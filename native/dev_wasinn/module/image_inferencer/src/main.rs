use std::{fs, sync::OnceLock};
use log::{error, warn, info};

use inferencer::MobilnetModel;

static MODEL: OnceLock<MobilnetModel> = OnceLock::new();

fn main() {
    // Initialize logger for WASM
    #[cfg(target_arch = "wasm32")]
    wasm_logger::init(wasm_logger::Config::default());
    
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    
    info!("Inferencer initialized");
}

fn load_model_fs() -> Result<(), String> {
    let xml = fs::read_to_string("fixture/model.xml")
        .map_err(|e| {
            let err_msg = format!("Failed to read model.xml: {}", e);
            error!("{}", err_msg);
            err_msg
        })?;
    let weights = fs::read("fixture/model.bin")
        .map_err(|e| {
            let err_msg = format!("Failed to read model.bin: {}", e);
            error!("{}", err_msg);
            err_msg
        })?;
    
    let model = MobilnetModel::from_buffer_result(xml.into_bytes(), weights)
        .map_err(|e| {
            error!("Failed to create model from filesystem: {}", e);
            e
        })?;
    
    MODEL.set(model)
        .map_err(|_| {
            let err_msg = "Model already loaded";
            warn!("{}", err_msg);
            err_msg.to_string()
        })?;
    
    info!("Model loaded from filesystem successfully");
    Ok(())
}

#[no_mangle]
pub extern "C" fn load_model(xml_ptr: i32, xml_len: i32, weights_ptr: i32, weights_len: i32) -> i32 {
    // Basic validation
    if xml_ptr == 0 || xml_len <= 0 || weights_ptr == 0 || weights_len <= 0 {
        error!("Invalid parameters: xml_ptr={}, xml_len={}, weights_ptr={}, weights_len={}", 
               xml_ptr, xml_len, weights_ptr, weights_len);
        return -1;
    }
    
    let xml = unsafe { std::slice::from_raw_parts(xml_ptr as *const u8, xml_len as usize) };
    let weights = unsafe { std::slice::from_raw_parts(weights_ptr as *const u8, weights_len as usize) };
    
    match MobilnetModel::from_buffer_result(xml.to_vec(), weights.to_vec()) {
        Ok(model) => {
            match MODEL.set(model) {
                Ok(_) => {
                    info!("Model loaded successfully via FFI");
                    0 // Success
                }
                Err(_) => {
                    warn!("Model already loaded");
                    1 // Already loaded
                }
            }
        }
        Err(e) => {
            error!("Failed to load model: {}", e);
            -1 // Error
        }
    }
}

#[no_mangle]
pub extern "C" fn infer(tensor_ptr: i32, tensor_len: i32, result_ptr: i32) -> i32 {
    // Basic validation
    if tensor_ptr == 0 || tensor_len <= 0 || result_ptr == 0 {
        error!("Invalid parameters: tensor_ptr={}, tensor_len={}, result_ptr={}", 
               tensor_ptr, tensor_len, result_ptr);
        return -1;
    }
    
    let tensor_raw = unsafe { std::slice::from_raw_parts(tensor_ptr as *const u8, tensor_len as usize) };
    
    let model = match MODEL.get() {
        Some(m) => m,
        None => {
            warn!("Model not loaded, attempting to load from filesystem");
            if let Err(e) = load_model_fs() {
                error!("Failed to load model from filesystem: {}", e);
                return -2; // Model load failed
            }
            MODEL.get().unwrap()
        }
    };
    
    let tensor = match model.tensor_from_raw_data(tensor_raw) {
        Ok(t) => t,
        Err(e) => {
            error!("Invalid tensor data: {:?}", e);
            return -3; // Invalid tensor
        }
    };
    
    let (label, confidence) = match model.run_inference_compat(tensor) {
        Ok(Some(r)) => (r.0 as u32, r.1),
        Ok(None) => {
            warn!("No inference result returned");
            (0, 0.0)
        }
        Err(e) => {
            error!("Inference failed: {}", e);
            return -4; // Inference failed
        }
    };
    
    unsafe {
        let result_u8_ptr = result_ptr as *mut u8;
        std::ptr::write(result_u8_ptr as *mut u32, label);
        std::ptr::write(result_u8_ptr.add(4) as *mut f32, confidence);
    }
    
    0 // Success
}
