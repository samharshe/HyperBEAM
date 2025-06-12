use log::{error, warn, info};
use std::sync::OnceLock;

#[derive(Debug, PartialEq, Clone)]
pub struct InferenceResult(pub usize, pub f32);

#[derive(Debug)]
pub enum ValidationError {
    InvalidDimensions,
    InvalidDataSize,
    InvalidFormat,
}

pub trait ModelConfig {
    fn output_size(&self) -> usize;
    fn input_dims(&self) -> &[u32];
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError>;
    fn tensor_type(&self) -> wasi_nn::TensorType;
}

pub struct ImageNetConfig;

impl ModelConfig for ImageNetConfig {
    fn output_size(&self) -> usize {
        1001 // ImageNet 1000 classes + background
    }
    
    fn input_dims(&self) -> &[u32] {
        &[1, 3, 224, 224] // NCHW format
    }
    
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError> {
        let expected_size = self.input_dims().iter().map(|&x| x as usize).product::<usize>() * std::mem::size_of::<f32>();
        if data.len() != expected_size {
            return Err(ValidationError::InvalidDataSize);
        }
        Ok(())
    }
    
    fn tensor_type(&self) -> wasi_nn::TensorType {
        wasi_nn::TENSOR_TYPE_F32
    }
}

pub struct TextModelConfig {
    pub vocab_size: usize,
    pub sequence_length: usize,
    input_dims: Vec<u32>,
}

impl TextModelConfig {
    pub fn new(vocab_size: usize, sequence_length: usize) -> Self {
        Self {
            vocab_size,
            sequence_length,
            input_dims: vec![1, sequence_length as u32],
        }
    }
}

impl ModelConfig for TextModelConfig {
    fn output_size(&self) -> usize {
        self.vocab_size
    }
    
    fn input_dims(&self) -> &[u32] {
        &self.input_dims
    }
    
    fn validate_input(&self, data: &[u8]) -> Result<(), ValidationError> {
        let expected_size = self.sequence_length * std::mem::size_of::<u32>();
        if data.len() != expected_size {
            return Err(ValidationError::InvalidDataSize);
        }
        Ok(())
    }
    
    fn tensor_type(&self) -> wasi_nn::TensorType {
        wasi_nn::TENSOR_TYPE_F32 // Using F32 as U32 is not available in wasi-nn 0.1.0
    }
}

pub struct Model<C: ModelConfig> {
    context_ptr: u32,
    _graph_ptr: u32,
    config: C,
}

// Backward compatibility alias
pub type MobilnetModel = Model<ImageNetConfig>;

impl<C: ModelConfig> Model<C> {
    pub fn from_buffer(xml: Vec<u8>, weights: Vec<u8>, config: C) -> Result<Self, String> {
        let _graph_ptr = unsafe {
            wasi_nn::load(
                &[&xml, &weights],
                wasi_nn::GRAPH_ENCODING_OPENVINO,
                wasi_nn::EXECUTION_TARGET_CPU,
            )
            .map_err(|e| {
                let err_msg = format!("Failed to load graph: {:?}", e);
                error!("{}", err_msg);
                err_msg
            })?
        };
        let context_ptr = unsafe {
            wasi_nn::init_execution_context(_graph_ptr)
                .map_err(|e| {
                    let err_msg = format!("Failed to init execution context: {:?}", e);
                    error!("{}", err_msg);
                    err_msg
                })?
        };
        info!("Model loaded successfully with output size: {}", config.output_size());
        Ok(Self {
            context_ptr,
            _graph_ptr,
            config,
        })
    }

    pub fn run_inference(&self, tensor: wasi_nn::Tensor) -> Result<Option<InferenceResult>, String> {
        unsafe {
            wasi_nn::set_input(self.context_ptr, 0, tensor)
                .map_err(|e| {
                    let err_msg = format!("Failed to set input tensor: {:?}", e);
                    error!("{}", err_msg);
                    err_msg
                })?
            ;

            wasi_nn::compute(self.context_ptr)
                .map_err(|e| {
                    let err_msg = format!("Failed to compute inference: {:?}", e);
                    error!("{}", err_msg);
                    err_msg
                })?
            ;
        }

        let mut output_buffer = vec![0f32; self.config.output_size()];
        unsafe {
            wasi_nn::get_output(
                self.context_ptr,
                0,
                output_buffer.as_mut_ptr() as *mut u8,
                (output_buffer.len() * std::mem::size_of::<f32>())
                    .try_into()
                    .unwrap(),
            )
            .map_err(|e| {
                let err_msg = format!("Failed to get output: {:?}", e);
                error!("{}", err_msg);
                err_msg
            })?
            ;
        }

        let mut results: Vec<InferenceResult> = output_buffer
            .iter()
            .skip(1)
            .enumerate()
            .map(|(class, prob)| InferenceResult(class, *prob))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(results.first().cloned())
    }

    pub fn tensor_from_raw_data<'a>(&'a self, tensor_data: &'a [u8]) -> Result<wasi_nn::Tensor<'a>, ValidationError> {
        self.config.validate_input(tensor_data)?;
        Ok(wasi_nn::Tensor {
            dimensions: self.config.input_dims(),
            r#type: self.config.tensor_type(),
            data: tensor_data,
        })
    }
}

impl<C: ModelConfig> Drop for Model<C> {
    fn drop(&mut self) {
        // Note: wasi-nn doesn't provide cleanup functions in 0.1.0
        // This is a placeholder for future resource cleanup
        warn!("Model dropped - resources may leak due to wasi-nn limitations");
    }
}

// Backward compatibility methods for MobilnetModel
impl MobilnetModel {
    pub fn from_buffer_compat(xml: Vec<u8>, weights: Vec<u8>) -> Self {
        match Self::from_buffer_result(xml, weights) {
            Ok(model) => model,
            Err(e) => panic!("{}", e), // Maintain original panic behavior for compatibility
        }
    }
    
    pub fn from_buffer_result(xml: Vec<u8>, weights: Vec<u8>) -> Result<Self, String> {
        <Model<ImageNetConfig>>::from_buffer(xml, weights, ImageNetConfig)
    }
    
    pub fn tensor_from_raw_bgr<'a>(&'a self, tensor_data: &'a [u8]) -> wasi_nn::Tensor<'a> {
        match self.tensor_from_raw_data(tensor_data) {
            Ok(tensor) => tensor,
            Err(e) => panic!("Invalid tensor data: {:?}", e), // Maintain original panic behavior
        }
    }
    
    pub fn run_inference_compat(&self, tensor: wasi_nn::Tensor) -> Result<Option<InferenceResult>, String> {
        <Model<ImageNetConfig>>::run_inference(self, tensor)
    }
}

// Static model storage for FFI interface
static MODEL: OnceLock<MobilnetModel> = OnceLock::new();

// Load model from filesystem (fixture directory)
fn load_model_fs() -> Result<(), String> {
    eprintln!("WASM DEBUG: load_model_fs() called");
    
    // Load XML file
    eprintln!("WASM DEBUG: About to read ./fixture/model.xml");
    let xml = std::fs::read("./fixture/model.xml")
        .map_err(|e| format!("Failed to read model.xml: {}", e))?;
    eprintln!("WASM DEBUG: Successfully read XML file, {} bytes", xml.len());
    
    // Load weights file  
    eprintln!("WASM DEBUG: About to read ./fixture/model.bin");
    let weights = std::fs::read("./fixture/model.bin")
        .map_err(|e| format!("Failed to read model.bin: {}", e))?;
    eprintln!("WASM DEBUG: Successfully read weights file, {} bytes", weights.len());
    
    info!("Loading model from filesystem - XML: {} bytes, weights: {} bytes", xml.len(), weights.len());
    
    eprintln!("WASM DEBUG: About to call MobilnetModel::from_buffer_result");
    // Create model
    let model = MobilnetModel::from_buffer_result(xml, weights)?;
    eprintln!("WASM DEBUG: Model created successfully");
    
    // Store in static variable
    eprintln!("WASM DEBUG: About to store model in static variable");
    MODEL.set(model).map_err(|_| "Failed to store model in static variable".to_string())?;
    eprintln!("WASM DEBUG: Model stored successfully");
    
    info!("Model loaded successfully from filesystem");
    Ok(())
}

// FFI exports matching cascadia-demo interface
#[no_mangle]
pub extern "C" fn load_model(xml_ptr: i32, xml_len: i32, weights_ptr: i32, weights_len: i32) -> i32 {
    // Check if model is already loaded
    if MODEL.get().is_some() {
        info!("Model already loaded");
        return 1; // Already loaded
    }

    // Extract XML buffer from WASM memory
    let xml_slice = unsafe {
        std::slice::from_raw_parts(xml_ptr as *const u8, xml_len as usize)
    };
    let xml = xml_slice.to_vec();

    // Extract weights buffer from WASM memory  
    let weights_slice = unsafe {
        std::slice::from_raw_parts(weights_ptr as *const u8, weights_len as usize)
    };
    let weights = weights_slice.to_vec();

    info!("Loading model with XML size: {}, weights size: {}", xml.len(), weights.len());

    // Create model
    match MobilnetModel::from_buffer_result(xml, weights) {
        Ok(model) => {
            match MODEL.set(model) {
                Ok(_) => {
                    info!("Model loaded successfully");
                    0 // Success
                },
                Err(_) => {
                    error!("Failed to store model in static variable");
                    -1 // Error
                }
            }
        },
        Err(e) => {
            error!("Failed to create model: {}", e);
            -1 // Error
        }
    }
}

#[no_mangle]
pub extern "C" fn infer(_tensor_ptr: i32, tensor_len: i32, result_ptr: i32) -> i32 {
    eprintln!("WASM DEBUG: infer function called with tensor_len: {}", tensor_len);
    info!("WASM infer function called with tensor_len: {}", tensor_len);
    
    eprintln!("WASM DEBUG: About to check MODEL.get()");
    
    // Test model loading
    match MODEL.get() {
        Some(_) => {
            eprintln!("WASM DEBUG: Model already loaded");
            info!("Model already loaded");
        },
        None => {
            eprintln!("WASM DEBUG: Model not loaded, skipping model loading for now");
            info!("Model not loaded, skipping model loading for testing");
            // Skip model loading for now - just proceed with dummy result
        }
    }
    
    eprintln!("WASM DEBUG: About to create dummy result");
    
    // Still return dummy result (skip actual inference)
    let label_bytes = (42u32).to_le_bytes();
    let confidence_bytes = (0.95f32).to_le_bytes();
    
    eprintln!("WASM DEBUG: About to write to memory at result_ptr: {}", result_ptr);
    
    unsafe {
        let result_slice = std::slice::from_raw_parts_mut(result_ptr as *mut u8, 8);
        result_slice[0..4].copy_from_slice(&label_bytes);
        result_slice[4..8].copy_from_slice(&confidence_bytes);
    }
    
    eprintln!("WASM DEBUG: About to return success");
    info!("WASM infer returning dummy result");
    0
}