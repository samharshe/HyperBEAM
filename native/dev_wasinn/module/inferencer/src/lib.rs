mod bindings;
mod components;
mod utils;

use std::{cell::RefCell, collections::HashMap};

use bindings::{exports::ncl::ml::chatbot, ncl::ml::token_generator};
use components::wasi::nn::{
    graph::{load_by_name, GraphExecutionContext},
    tensor::{Tensor, TensorType},
};
use utils::{bytes_slice_to_f32_slice, get_attention_mask, i64_slice_to_bytes_slice, softmax};

thread_local! {
    static CHATBOT: RefCell<Chatbot> = RefCell::new(Chatbot::default());
}

struct ChatbotSession
{
    execution_context: GraphExecutionContext,
    config: chatbot::SessionConfig,
    id: u64,
}
struct Chatbot
{
    sessions: HashMap<u64, ChatbotSession>,
    id_counter: u64,
}
impl Default for Chatbot
{
    fn default() -> Self
    {
        Self {
            sessions: HashMap::new(),
            id_counter: 0,
        }
    }
}

impl chatbot::Guest for Chatbot
{
    fn infer(session_id: chatbot::SessionId, ids: Vec<i64>) -> Result<(), chatbot::Errors>
    {
        CHATBOT.with_borrow_mut(|chatbot| {
            let session = chatbot.sessions.get(&session_id).ok_or(chatbot::Errors::InvalidSession)?;
            let mut length = ids.len() as u32;
            let raw_tensor_length = (length as usize) * std::mem::size_of::<i64>();
            let tokens_dims = &mut [1u32, length];

            // prepare input tensors
            // input_ids
            let mut input_ids_raw = vec![0u8; ids.len() * std::mem::size_of::<i64>()];
            i64_slice_to_bytes_slice(&ids, &mut input_ids_raw);
            // position_ids
            let mut position_ids_raw = Vec::with_capacity(raw_tensor_length);
            for i in 0..length {
                position_ids_raw.extend_from_slice(&(i as i64).to_le_bytes());
            }
            // attention_mask
            let mut attention_mask_raw = Vec::with_capacity(raw_tensor_length);
            for &id in &ids {
                attention_mask_raw.extend_from_slice(&get_attention_mask(&id).to_le_bytes());
            }

            let mut count = 0;
            loop {
                println!("count {}, max: {:?}", count, session.config.max_token);
                match session.config.max_token {
                    Some(i) if count >= i => break,
                    _ => (),
                };

                let input_ids = Tensor::new(tokens_dims, TensorType::I64, &input_ids_raw);
                let position_ids = Tensor::new(tokens_dims, TensorType::I64, &position_ids_raw);
                let attention_mask = Tensor::new(tokens_dims, TensorType::I64, &attention_mask_raw);

                // set inputs on the host
                session.execution_context.set_input("input_ids", input_ids).unwrap();
                session.execution_context.set_input("position_ids", position_ids).unwrap();
                session.execution_context.set_input("attention_mask", attention_mask).unwrap();

                // run inference on the host
                session.execution_context.compute().unwrap();

                // get output from host
                let output = session.execution_context.get_output("logits").unwrap();
                let logits_raw = output.data();
                let dimensions = output.dimensions();
                let [_, seq_len, vocab_size] = match dimensions[..3] {
                    [a, b, c] => [a, b, c],
                    _ => panic!("tensor dimensions must match [batch_len, seq_len, vocab_size]"),
                };

                // slice the next logit
                let next_logit_start: usize = ((seq_len - 1) * vocab_size) as usize;
                let next_logit_end: usize = next_logit_start + (vocab_size as usize);
                let byte_start = next_logit_start * std::mem::size_of::<f32>();
                let byte_end = next_logit_end * std::mem::size_of::<f32>();
                let next_logit_bytes: &[u8] = &logits_raw[byte_start..byte_end];
                // TODO: vocab_size should be known prior to inference, we can reuse this buffer and avoid re-allocation on stack.
                let mut next_logit = vec![0f32; vocab_size as usize];
                bytes_slice_to_f32_slice(next_logit_bytes, &mut next_logit);

                // softmax to get predictions
                softmax(&mut next_logit);

                // pick max prediction
                let (next_token, _) = next_logit.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();

                // check for end-of-sequence token
                match next_token {
                    128000..=128255 => break,
                    _ => match token_generator::generate(session.id, next_token as u32) {
                        1 => {
                            // extend input slices to include next_token
                            input_ids_raw.extend_from_slice(&(next_token as i64).to_le_bytes());
                            position_ids_raw.extend_from_slice(&(length as i64).to_le_bytes());
                            attention_mask_raw
                                .extend_from_slice(&get_attention_mask(&(next_token as i64)).to_le_bytes());
                            length += 1;
                            tokens_dims[1] = length;
                            count += 1;
                        },
                        _ => break,
                    },
                }
            }
            Ok(())
        })
    }
    fn register(config: chatbot::SessionConfig) -> chatbot::SessionId
    {
        CHATBOT.with_borrow_mut(|chatbot| {
            let graph = load_by_name(&config.model_id).unwrap();
            let execution_context = graph.init_execution_context().unwrap();

            let id = chatbot.id_counter;
            chatbot.sessions.insert(
                id,
                ChatbotSession {
                    execution_context,
                    config,
                    id: chatbot.id_counter.clone(),
                },
            );
            chatbot.id_counter = id + 1;
            id
        })
    }
}

bindings::export!(Chatbot with_types_in bindings);
