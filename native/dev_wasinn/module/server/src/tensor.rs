use image::{DynamicImage, ImageReader, RgbImage};
use opencv::core::{MatTraitConst, Scalar_};
use tokio::sync::broadcast::Sender as LogSender;

pub fn jpeg_to_openvino_tensor(jpeg_bytes: Vec<u8>, log_sender: &LogSender<String>) -> anyhow::Result<Vec<u8>>
{
    log_sender.send("[server/tensor.rs] Starting JPEG to BGR conversion.".to_string()).ok();

    let buf = opencv::core::Mat::from_slice(&jpeg_bytes)?;
    let jpeg = opencv::imgcodecs::imdecode(&buf, opencv::imgcodecs::IMREAD_COLOR)?;
    log_sender.send("[server/tensor.rs] Successfully decoded JPEG image.".to_string()).ok();

    let mut resized =
        opencv::core::Mat::new_rows_cols_with_default(224, 224, opencv::core::CV_32FC3, Scalar_::all(0.0))?;
    let dst_size = resized.size()?;
    opencv::imgproc::resize(&jpeg, &mut resized, dst_size, 0.0, 0.0, opencv::imgproc::INTER_LINEAR)?;
    log_sender.send("[server/tensor.rs] Resized image to 224x224.".to_string()).ok();

    let mut dst = opencv::core::Mat::new_rows_cols_with_default(224, 224, opencv::core::CV_32FC3, Scalar_::all(0.0))?;
    resized.convert_to(&mut dst, opencv::core::CV_32FC3, 1.0, 0.0)?;
    log_sender.send("[server/tensor.rs] Converted image to float32 format.".to_string()).ok();

    let mut nchw_data = vec![0f32; 224 * 224 * 3];

    for h in 0..224 {
        for w in 0..224 {
            let pixel_idx = h * 224 + w;

            let pixel = dst.at_2d::<opencv::core::Vec3f>(h, w)?;
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
        std::slice::from_raw_parts(nchw_data.as_ptr() as *const u8, nchw_data.len() * std::mem::size_of::<f32>())
    };

    log_sender.send("[server/tensor.rs] Successfully converted image to raw bytes.".to_string()).ok();
    Ok(bytes.to_vec())
}

pub fn jpeg_to_squeezenet_tensor(jpeg_bytes: Vec<u8>) -> Vec<u8>
{
    let cursor = std::io::Cursor::new(jpeg_bytes);
    let mut pixels = ImageReader::new(cursor);
    pixels.set_format(image::ImageFormat::Jpeg);
    let pixels = pixels.decode().unwrap();
    let dyn_img: DynamicImage = pixels.resize_exact(224, 224, image::imageops::Triangle);
    let bgr_img: RgbImage = dyn_img.to_rgb8();

    // Get an array of the pixel values
    let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];

    // Create an array to hold the f32 value of those pixels
    let bytes_required = raw_u8_arr.len() * 4;
    let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

    // Normalizing values for the model
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    // Read the number as a f32 and break it into u8 bytes
    for i in 0..raw_u8_arr.len() {
        let u8_f32: f32 = raw_u8_arr[i] as f32;
        let rgb_iter = i % 3;

        // Normalize the pixel
        let norm_u8_f32: f32 = (u8_f32 / 255.0 - mean[rgb_iter]) / std[rgb_iter];

        // Convert it to u8 bytes and write it with new shape
        let u8_bytes = norm_u8_f32.to_ne_bytes();
        for j in 0..4 {
            u8_f32_arr[(raw_u8_arr.len() * 4 * rgb_iter / 3) + (i / 3) * 4 + j] = u8_bytes[j];
        }
    }

    return u8_f32_arr;
}
