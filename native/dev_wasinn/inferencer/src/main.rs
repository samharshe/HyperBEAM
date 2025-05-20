use std::{fs, sync::OnceLock};

use inferencer::MobilnetModel;

static MODEL: OnceLock<MobilnetModel> = OnceLock::new();

fn main() {}

fn load_model_fs() {
    let xml = fs::read_to_string("fixture/model.xml").unwrap();
    let weights = fs::read("fixture/model.bin").unwrap();
    MODEL
        .set(MobilnetModel::from_buffer(xml.into_bytes(), weights))
        .ok();
}

#[no_mangle]
pub extern "C" fn load_model(xml_ptr: i32, xml_len: i32, weights_ptr: i32, weights_len: i32) {
    let xml = unsafe { std::slice::from_raw_parts(xml_ptr as *const u8, xml_len as usize) };
    let weights =
        unsafe { std::slice::from_raw_parts(weights_ptr as *const u8, weights_len as usize) };
    MODEL
        .set(MobilnetModel::from_buffer(xml.to_vec(), weights.to_vec()))
        .ok();
}

#[no_mangle]
pub extern "C" fn infer(tensor_ptr: i32, tensor_len: i32, result_ptr: i32) {
    let tensor_raw =
        unsafe { std::slice::from_raw_parts(tensor_ptr as *const u8, tensor_len as usize) };
    let model = match MODEL.get() {
        Some(m) => m,
        None => {
            load_model_fs();
            MODEL.get().unwrap()
        }
    };
    let tensor = model.tensor_from_raw_bgr(tensor_raw);
    let (label, confidence) = match model.run_inference(tensor) {
        Some(r) => (r.0 as u32, r.1),
        None => (0, 0.0),
    };
    unsafe {
        let result_u8_ptr = result_ptr as *mut u8;
        std::ptr::write(result_u8_ptr as *mut u32, label);
        std::ptr::write(result_u8_ptr.add(4) as *mut f32, confidence);
    }
}
