use opencv::core::{MatTraitConst, MatTrait, Scalar_};
use tokio::sync::broadcast::Sender as LogSender;

pub fn jpeg_to_raw_bgr(jpeg_bytes: Vec<u8>, log_sender: &LogSender<String>) -> anyhow::Result<Vec<u8>> {
    log_sender.send("[server/tensor.rs] Starting JPEG to BGR conversion.".to_string()).ok();
    
    let buf = opencv::core::Mat::from_slice(&jpeg_bytes)?;
    let jpeg = opencv::imgcodecs::imdecode(&buf, opencv::imgcodecs::IMREAD_COLOR)?;
    log_sender.send("[server/tensor.rs] Successfully decoded JPEG image.".to_string()).ok();
    
    let mut resized = opencv::core::Mat::new_rows_cols_with_default(
        224, 224, 
        opencv::core::CV_32FC3, 
        Scalar_::all(0.0)
    )?;
    let dst_size = resized.size()?;
    opencv::imgproc::resize(&jpeg, &mut resized, dst_size, 0.0, 0.0, opencv::imgproc::INTER_LINEAR)?;
    log_sender.send("[server/tensor.rs] Resized image to 224x224.".to_string()).ok();
    
    let mut dst = opencv::core::Mat::new_rows_cols_with_default(
        224, 224, 
        opencv::core::CV_32FC3, 
        Scalar_::all(0.0)
    )?;
    resized.convert_to(&mut dst, opencv::core::CV_32FC3, 1.0, 0.0)?;
    log_sender.send("[server/tensor.rs] Converted image to float32 format.".to_string()).ok();
    
    let mut nchw_data = vec![0f32; 224 * 224 * 3];
    
    for h in 0..224 {
        for w in 0..224 {
            let pixel_idx = h * 224 + w;
            
            let pixel = dst.at_2d_mut::<opencv::core::Vec3f>(h, w)?;
            let b = pixel[0];
            let g = pixel[1];
            let r = pixel[2];
            
            nchw_data[pixel_idx as usize] = b;
            nchw_data[224 * 224 + pixel_idx as usize] = g;
            nchw_data[2 * 224 * 224 + pixel_idx as usize] = r;
        }
    }
    log_sender.send("[server/tensor.rs] Converted image to NCHW format.".to_string()).ok();
    
    let bytes = unsafe {
        std::slice::from_raw_parts(
            nchw_data.as_ptr() as *const u8,
            nchw_data.len() * std::mem::size_of::<f32>()
        )
    };
    
    log_sender.send("[server/tensor.rs] Successfully converted image to raw bytes.".to_string()).ok();
    Ok(bytes.to_vec())
}