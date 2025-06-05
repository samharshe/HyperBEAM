use std::{
    io::{self, BufRead, Write},
    path::Path,
    sync::Arc,
};

use anyhow::Result;
use server::runtime::WasmInstance;
use tokenizers::tokenizer::Tokenizer;
use wasmtime::{component::Component, Config, Engine};

fn main() -> Result<()>
{
    tracing_subscriber::fmt::Subscriber::builder()
        .with_max_level(tracing::Level::DEBUG)
        .with_env_filter("debug")
        .init();

    let tokenizer_path = "./models/onnx/llama3.1-8b-instruct/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let engine = Arc::new(Engine::new(&Config::new()).unwrap());
    let module = Arc::new(
        Component::from_file(&engine, Path::new("../inferencer/target/wasm32-wasip1/release/ncl_ml.wasm")).unwrap(),
    );
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut instance = WasmInstance::new(engine.clone(), module.clone(), "llama3.1-8b-instruct")?;
    let (session_id, session_receiver) = instance.register().unwrap();
    println!("🦙 Chatbot ready. Type a message or 'exit':");
    for line in stdin.lock().lines() {
        let input = line?;
        if input.trim().to_lowercase() == "exit" {
            break;
        }

        let prompt = format!(r#"<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"#);
        let encoding = tokenizer.encode(prompt.clone(), false).unwrap();
        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        
        eprintln!("DEBUG main: Prompt: {}", prompt);
        
        // generate tokens autoregressively
        let max_tokens = 50;
        // I am not sure what the difference is; both of these seem intended to terminate token generation; we stop on either of them below
        let end_of_text = 128001;
        let eot_id = 128009;
        let mut generated_tokens = Vec::new();
        
        let mut instance = WasmInstance::new(engine.clone(), module.clone(), "llama3.1-8b-instruct")?;
        
        for i in 0..max_tokens {
            let result = instance.infer_llm(ids.clone())?;
            if result.is_empty() {
                eprintln!("ERROR: No token generated");
                break;
            }
            
            let next_token = result[0];
            eprintln!("DEBUG main: Generated token {} at position {}", next_token, i);
            
            // check for end-of-sequence token
            if next_token == end_of_text as u32 || next_token == eot_id as u32 {
                eprintln!("DEBUG main: end-of-sequence token generated; stopping.");
                break;
            }
            
            // add the generated token to our sequence
            ids.push(next_token as i64);
            generated_tokens.push(next_token);
        }
        
        eprintln!("DEBUG main: Total generated tokens: {}", generated_tokens.len());
        eprintln!("DEBUG main: Generated tokens: {:?}", &generated_tokens);
        
        let response = tokenizer.decode(&generated_tokens, false).unwrap();
        writeln!(stdout, "Bot: {}\n", response)?;
        stdout.flush()?;
    }

    println!("👋 Goodbye.");
    Ok(())
}
