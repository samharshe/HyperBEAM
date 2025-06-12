use tokenizers::Tokenizer;
use tokenizers::tokenizer::Result as TokenizerResult;
use tokenizers::utils::padding::{PaddingParams, PaddingDirection};

fn load_tokenizer(path: &str) -> TokenizerResult<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path)?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: tokenizers::utils::padding::PaddingStrategy::Fixed(512),
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: Some("<pad>".to_string()),
    }));
    Ok(tokenizer)
}


fn prepare_tensors(
    tokenizer: &Tokenizer,
    prompt: &str,
    max_len: usize,
) -> Result<(wasi_nn::Tensor, wasi_nn::Tensor)> {
    let encoding = tokenizer.encode(prompt, true).map_err(|e| {
        ChatbotError::Other(format!("Tokenization error: {}", e))
    })?;

    let mut ids = encoding.get_ids().to_vec();
    if ids.len() > max_len {
        ids.truncate(max_len);
    } else {
        ids.resize(max_len, 0); // pad with 0 (pad_id)
    }

    let positions: Vec<i32> = (0..ids.len()).map(|i| i as i32).collect();

    let tokens_tensor = wasi_nn::Tensor {
        dimensions: &[1, max_len as u32],
        r#type: wasi_nn::TENSOR_TYPE_I32,
        data: cast_slice(&ids),
    };

    let pos_tensor = wasi_nn::Tensor {
        dimensions: &[1, max_len as u32],
        r#type: wasi_nn::TENSOR_TYPE_I32,
        data: cast_slice(&positions),
    };

    Ok((tokens_tensor, pos_tensor))
}
