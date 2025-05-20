#[derive(Debug, PartialEq, Clone)]
pub struct InferenceResult(pub usize, pub f32);
pub struct MobilnetModel {
    context_ptr: u32,
    _graph_ptr: u32,
}

impl MobilnetModel {
    pub fn from_buffer(xml: Vec<u8>, weights: Vec<u8>) -> Self {
        let _graph_ptr = unsafe {
            wasi_nn::load(
                &[&xml, &weights],
                wasi_nn::GRAPH_ENCODING_OPENVINO,
                wasi_nn::EXECUTION_TARGET_GPU,
            )
            .expect("Failed to load graph")
        };
        let context_ptr = unsafe {
            wasi_nn::init_execution_context(_graph_ptr).expect("Failed to init execution context")
        };
        Self {
            context_ptr,
            _graph_ptr,
        }
    }

    pub fn run_inference(&self, tensor: wasi_nn::Tensor) -> Option<InferenceResult> {
        unsafe {
            wasi_nn::set_input(self.context_ptr, 0, tensor).expect("Failed to set input tensor");

            wasi_nn::compute(self.context_ptr).expect("Failed to compute inference");
        }

        let mut output_buffer = vec![0f32; 1001];
        unsafe {
            wasi_nn::get_output(
                self.context_ptr,
                0,
                output_buffer.as_mut_ptr() as *mut u8,
                (output_buffer.len() * std::mem::size_of::<f32>())
                    .try_into()
                    .unwrap(),
            )
            .expect("Failed to get output");
        }

        let mut results: Vec<InferenceResult> = output_buffer
            .iter()
            .skip(1)
            .enumerate()
            .map(|(class, prob)| InferenceResult(class, *prob))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.first().cloned()
    }

    pub fn tensor_from_raw_bgr<'a>(&self, tensor_data: &'a [u8]) -> wasi_nn::Tensor<'a> {
        wasi_nn::Tensor {
            dimensions: &[1, 3, 224, 224],
            r#type: wasi_nn::TENSOR_TYPE_F32,
            data: tensor_data,
        }
    }
}
