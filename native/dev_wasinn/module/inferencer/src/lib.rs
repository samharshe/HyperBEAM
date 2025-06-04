wit_bindgen::generate!({
    path: "../../wasmtime/crates/wasi-nn/wit",
    world: "ml",
});

use self::wasi::nn::{
    graph::{load_by_name, Graph},
    tensor::{Tensor, TensorDimensions, TensorType},
};
mod bindings;
use std::sync::OnceLock;

use bindings::exports::component::inferer::mobilenet::Guest;
use ndarray::{Array, Dim};
static MODEL: OnceLock<MobilnetModel> = OnceLock::new();

#[derive(Debug)]
pub struct MobilnetModel
{
    graph: Graph,
}

impl Guest for MobilnetModel
{
    fn infer(registry_id: String, tensor: Vec<u8>) -> bindings::exports::component::inferer::mobilenet::InferResult
    {
        let model = match MODEL.get() {
            Some(m) => m,
            None => {
                let graph = load_by_name(&registry_id).unwrap();
                MODEL
                    .set(Self {
                        graph,
                    })
                    .ok();
                MODEL.get().unwrap()
            },
        };
        let context = model.graph.init_execution_context().unwrap();
        let dimensions: TensorDimensions = vec![1, 3, 224, 224];
        let t = Tensor::new(&dimensions, TensorType::Fp32, &tensor);
        context.set_input("data", t).unwrap();
        context.compute().unwrap();
        let output_data = context.get_output("squeezenet0_flatten0_reshape0").unwrap().data();
        let output_f32 = bytes_to_f32_vec(output_data);

        let output_shape = [1, 1000, 1, 1];
        let output_tensor = Array::from_shape_vec(output_shape, output_f32).unwrap();
        let exp_output = output_tensor.mapv(|x| x.exp());
        let sum_exp_output = exp_output.sum_axis(ndarray::Axis(1));
        let softmax_output = exp_output / &sum_exp_output;

        let mut sorted = softmax_output
            .axis_iter(ndarray::Axis(1))
            .enumerate()
            .into_iter()
            .map(|(i, v)| (i, v[Dim([0, 0, 0])]))
            .collect::<Vec<(_, _)>>();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let (label, confidence) = sorted[0];
        (label as u32, confidence)
    }

    fn infer_llm(registry_id: String, ids: Vec<i64>) -> Vec<u32>
    {
        let model = match MODEL.get() {
            Some(m) => m,
            None => {
                let graph = load_by_name(&registry_id).unwrap();
                MODEL
                    .set(Self {
                        graph,
                    })
                    .ok();
                MODEL.get().unwrap()
            },
        };
        let length = ids.len();
        let tokens_dims = &[1u32, length as u32];
        let i = i64_vec_to_bytes(ids.clone());
        let tokens_tensor = Tensor::new(tokens_dims, TensorType::I64, &i);

        let input_pos: Vec<i64> = (0..length as i64).collect();
        let i = i64_vec_to_bytes(input_pos);
        let input_pos_tensor = Tensor::new(tokens_dims, TensorType::I64, &i);

        let mut am: Vec<i64> = ids
            .iter()
            .map(|&id| {
                match id {
                    128000..=128255 => 0,
                    _ => 1
                }
            })
            .collect();

        let i = i64_vec_to_bytes(am);
        let am_tensor = Tensor::new(tokens_dims, TensorType::I64, &i);

        let context = model.graph.init_execution_context().unwrap();
        context.set_input("input_ids", tokens_tensor).unwrap();
        context.set_input("position_ids", input_pos_tensor).unwrap();
        context.set_input("attention_mask", am_tensor).unwrap();
        context.compute().unwrap();

        let output = context.get_output("logits").unwrap();
        let data = output.data();
        let dims = output.dimensions();
        let batch_len = dims[0];
        let seq_len = dims[1];
        let vocab_size = dims[2];
        
        let mut logits_f32: Vec<f32> = bytes_to_f32_vec(data);

        
        // only process the last position's logits for autoregressive generation
        let last_pos = seq_len - 1;
        let start = (last_pos * vocab_size) as usize;
        let end = start + (vocab_size as usize);
        let mut logits = &mut logits_f32[start..end];
        
        softmax(logits);

        let (token_id, prob) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        
        eprintln!("DEBUG: Selected token ID {} with probability {}", token_id, prob);
        
        // return only the single next token
        vec![token_id as u32]
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

#[derive(Debug, PartialEq, Clone)]
pub struct InferenceResult(pub usize, pub f32);

pub fn i64_vec_to_bytes(data: Vec<i64>) -> Vec<u8>
{
    let chunks: Vec<[u8; 8]> = data.into_iter().map(|f| f.to_le_bytes()).collect();
    let result: Vec<u8> = chunks.iter().flatten().copied().collect();
    result
}

pub fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32>
{
    let chunks: Vec<&[u8]> = data.chunks(4).collect();
    let v: Vec<f32> = chunks.into_iter().map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

    v.into_iter().collect()
}

bindings::export!(MobilnetModel with_types_in bindings);
