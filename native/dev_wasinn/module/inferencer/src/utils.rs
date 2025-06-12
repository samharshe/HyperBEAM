pub fn get_attention_mask(token_id: &i64) -> i64
{
    match token_id {
        128000..=128255 => 0,
        _ => 1,
    }
}

pub fn softmax(x: &mut [f32])
{
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

pub fn i64_slice_to_bytes_slice(input: &[i64], output: &mut [u8])
{
    assert_eq!(input.len() * std::mem::size_of::<i64>(), output.len(), "mismatch type sizes");

    for (i, val) in input.iter().enumerate() {
        let bytes = val.to_le_bytes();
        let start = i * std::mem::size_of::<i64>();
        output[start..start + std::mem::size_of::<i64>()].copy_from_slice(&bytes);
    }
}

pub fn bytes_slice_to_f32_slice(input: &[u8], output: &mut [f32])
{
    assert_eq!(input.len(), output.len() * std::mem::size_of::<f32>(), "mismatch type sizes");

    for (in_buf, out_buf) in input.chunks_exact(std::mem::size_of::<f32>()).zip(output.iter_mut()) {
        *out_buf = f32::from_le_bytes([in_buf[0], in_buf[1], in_buf[2], in_buf[3]]);
    }
}
