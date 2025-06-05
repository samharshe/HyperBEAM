mod bindings;

use bindings::exports::component::inferer::mobilenet::Guest;

wit_bindgen::generate!({
    path: "../../wasmtime/crates/wasi-nn/wit",
    world: "ml",
});

use self::wasi::nn::{
    graph::load_by_name,
    tensor::{Tensor, TensorType},
};

struct MobilnetModel;
impl Guest for MobilnetModel
{
    fn infer(registry_id: String, tensor: Vec<u8>) -> bindings::exports::component::inferer::mobilenet::InferResult
    {
        // let model = match MODEL.get() {
        //     Some(m) => m,
        //     None => {
        //         let graph = load_by_name(&registry_id).unwrap();
        //         MODEL
        //             .set(Self {
        //                 graph,
        //             })
        //             .ok();
        //         MODEL.get().unwrap()
        //     },
        // };
        // let context = model.graph.init_execution_context().unwrap();
        // let dimensions: TensorDimensions = vec![1, 3, 224, 224];
        // let t = Tensor::new(&dimensions, TensorType::Fp32, &tensor);
        // context.set_input("data", t).unwrap();
        // context.compute().unwrap();
        // let output_data = context.get_output("squeezenet0_flatten0_reshape0").unwrap().data();
        // let output_f32 = bytes_to_f32_vec(&output_data);

        // let output_shape = [1, 1000, 1, 1];
        // let output_tensor = Array::from_shape_vec(output_shape, output_f32).unwrap();
        // let exp_output = output_tensor.mapv(|x| x.exp());
        // let sum_exp_output = exp_output.sum_axis(ndarray::Axis(1));
        // let softmax_output = exp_output / &sum_exp_output;

        // let mut sorted = softmax_output
        //     .axis_iter(ndarray::Axis(1))
        //     .enumerate()
        //     .into_iter()
        //     .map(|(i, v)| (i, v[Dim([0, 0, 0])]))
        //     .collect::<Vec<(_, _)>>();
        // sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // let (label, confidence) = sorted[0];
        // (label as u32, confidence)
        (0, 0.0)
    }



    fn infer_llm(registry_id: String, ids: Vec<i64>) -> Vec<u32>
    {
        let graph = load_by_name(&registry_id).unwrap();
        let context = graph.init_execution_context().unwrap();
        
        let max_tokens = 20;
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
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

        for _ in 0..max_tokens {
            let input_ids = Tensor::new(tokens_dims, TensorType::I64, &input_ids_raw);
            let position_ids = Tensor::new(tokens_dims, TensorType::I64, &position_ids_raw);
            let attention_mask = Tensor::new(tokens_dims, TensorType::I64, &attention_mask_raw);
            
            // set inputs on the host
            context.set_input("input_ids", input_ids).unwrap();
            context.set_input("position_ids", position_ids).unwrap();
            context.set_input("attention_mask", attention_mask).unwrap();
            
            // run inference on the host
            context.compute().unwrap();
            
            // get output from host
            let output = context.get_output("logits").unwrap();
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
            let (next_token, _) = next_logit
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();

            // check for end-of-sequence token
            match next_token {
                128000..=128255 => break,
                _ => {
                    // extend input slices to include next_token
                    input_ids_raw.extend_from_slice(&(next_token as i64).to_le_bytes());
                    position_ids_raw.extend_from_slice(&(length as i64).to_le_bytes());
                    attention_mask_raw.extend_from_slice(&get_attention_mask(&(next_token as i64)).to_le_bytes());
                    length += 1;
                    tokens_dims[1] = length;
                    generated_tokens.push(next_token as u32);
                }
            }
        }   

        generated_tokens
    }
}

fn get_attention_mask(token_id: &i64) -> i64 {
    match token_id {
        128000..=128255 => 0,
        _ => 1
    }
}

pub fn softmax(x: &mut [f32]){
    let mut sum: f32 = 0.0;
    let mut max_val: f32 = x[0];

    for i in x.iter() {
        if *i > max_val {
            max_val = *i;
        }
    }

    for i in x.iter_mut() {
        *i = (*i - max_val).exp();
        sum += *i;
    }
    
    for i in x.iter_mut() {
        *i /= sum;
    } 
}

pub fn i64_slice_to_bytes_slice(input: &[i64], output: &mut [u8]) {
    assert_eq!(
        input.len() * std::mem::size_of::<i64>(),
        output.len(),
        "mismatch type sizes"
    );

    for (i, val) in input.iter().enumerate() {
        let bytes = val.to_le_bytes();
        let start = i * std::mem::size_of::<i64>();
        output[start..start + std::mem::size_of::<i64>()].copy_from_slice(&bytes);
    }
}

pub fn bytes_slice_to_f32_slice(input: &[u8], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len() * std::mem::size_of::<f32>(),
        "mismatch type sizes"
    );

    for (in_buf, out_buf) in input.chunks_exact(std::mem::size_of::<f32>()).zip(output.iter_mut()) {
        *out_buf = f32::from_le_bytes([in_buf[0], in_buf[1], in_buf[2], in_buf[3]]);
    }
}

bindings::export!(MobilnetModel with_types_in bindings);

#[derive(Debug, PartialEq, Clone)]
pub struct InferenceResult(pub usize, pub f32);
