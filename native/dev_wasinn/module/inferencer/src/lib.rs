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
}

#[derive(Debug, PartialEq, Clone)]
pub struct InferenceResult(pub usize, pub f32);

pub fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32>
{
    let chunks: Vec<&[u8]> = data.chunks(4).collect();
    let v: Vec<f32> = chunks.into_iter().map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

    v.into_iter().collect()
}

bindings::export!(MobilnetModel with_types_in bindings);
