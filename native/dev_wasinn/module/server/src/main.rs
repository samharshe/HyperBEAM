// pub fn main() -> anyhow::Result<()>
// {
//     tracing_subscriber::fmt::Subscriber::builder()
//         .with_max_level(tracing::Level::DEBUG)
//         .with_env_filter("debug")
//         .init();
//     let runtime = server::create_runtime()?;
//     runtime.block_on(server::start_server(3003, "../inferencer/target/wasm32-wasip1/release/inferencer.wasm".into()))
// }

use std::{
    io::{self, BufRead, Write},
    path::Path,
    sync::Arc,
};

use anyhow::Result;
use futures::stream::StreamExt;
use http_body_util::BodyExt;
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
        Component::from_file(&engine, Path::new("../inferencer/target/wasm32-wasip1/release/inferencer.wasm")).unwrap(),
    );

    println!("🦙 Chatbot ready. Type a message or 'exit':");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let input = line?;
        if input.trim().to_lowercase() == "exit" {
            break;
        }
        // prepare the tensor
        let prompt = format!(r#"<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
        {}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        "#, input);
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let ids = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mut instance = WasmInstance::new(engine.clone(), module.clone(), "llama3.1-8b-instruct")?;

        let result = instance.infer_llm(ids)?;
        let response = tokenizer.decode(&result, false).unwrap();
        writeln!(stdout, "Bot: {}\n", response)?;
        stdout.flush()?;
    }

    println!("👋 Goodbye.");
    Ok(())
}
