// pub fn main() -> anyhow::Result<()>
// {
//     tracing_subscriber::fmt::Subscriber::builder()
//         .with_max_level(tracing::Level::DEBUG)
//         .with_env_filter("debug")
//         .init();
//     let runtime = server::create_runtime()?;
//     runtime.block_on(server::start_server(3003, "../inferencer/target/wasm32-wasip1/release/inferencer.wasm".into()))
// }


use bytemuck::{cast_slice, cast_slice_mut};
use std::fs::File;
use std::io::{self, BufRead, Read, Write};
use std::process;
use std::{collections::HashMap, env};

use tokenizers::tokenizer::{Tokenizer, Result as TokenizerResult, TruncationParams};
use tokenizers::utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy};

use anyhow::Result;
use std::{net::SocketAddr, path::Path, sync::Arc};

use bytes::Buf;
use futures::stream::StreamExt;
use http_body_util::{BodyExt, StreamBody};
use hyper::{
    body::{Frame, Incoming as Body},
    header,
    server::conn::http1,
    service::service_fn,
    Method, Request, Response, StatusCode,
};
use hyper_util::rt::{TokioIo, TokioTimer};
use server::runtime::WasmInstance;
use tokio::{
    net::TcpListener,
    runtime::{Builder, Runtime},
    sync::{
        mpsc::{unbounded_channel, UnboundedSender},
        oneshot,
    },
    task::spawn_blocking,
};
use tokio_stream::wrappers::BroadcastStream;
use server::utils::{full, BoxBody, InferenceRequest};
use wasmtime::{component::Component, Config, Engine};



fn main() -> Result<()> {
    tracing_subscriber::fmt::Subscriber::builder()
        .with_max_level(tracing::Level::DEBUG)
        .with_env_filter("debug")
        .init();

    let tokenizer_path = "./models/onnx/llama3.1-8b-instruct/tokenizer.json";
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let engine = Arc::new(Engine::new(&Config::new()).unwrap());
    let module = Arc::new(Component::from_file(&engine, Path::new("../inferencer/target/wasm32-wasip1/release/inferencer.wasm")).unwrap());

    println!("🦙 Chatbot ready. Type a message or 'exit':");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    
    for line in stdin.lock().lines() {
        let input = line?;
        if input.trim().to_lowercase() == "exit" {
            break;
        }
        // prepare the tensor
        let prompt = r#"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

        What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        "#;
        let encoding = tokenizer.encode(prompt, true).unwrap();
        let ids = encoding.get_ids().iter().map(|&id| id as i64).collect();
        println!("Input IDs: {:?}", encoding.get_ids());

        let mut instance = WasmInstance::new(engine.clone(), module.clone(), "llama3.1-8b-instruct")?;

        let result = instance.infer_llm(ids)?;
        let response = "awaiting!";
        writeln!(stdout, "Bot: {}\n", response)?;
        stdout.flush()?;
    }

    println!("👋 Goodbye.");
    Ok(())
}


