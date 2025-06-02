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

struct LlamaChatbot {
    tokenizer: Tokenizer,
    max_length: usize,
}

impl LlamaChatbot {
    fn new(tokenizer_path: &str, max_length: usize) -> Result<Self> {
        let tokenizer = LlamaChatbot::load_tokenizer(tokenizer_path).unwrap();
        Ok(Self {
            tokenizer,
            max_length,
        })
    }

    fn load_tokenizer(path: &str) -> TokenizerResult<Tokenizer> {
        let mut tokenizer = Tokenizer::from_file(path)?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(512),
            pad_id: 128009,
            pad_type_id: 0,
            ..Default::default()
        }));
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: 512,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        }));
        Ok(tokenizer)
    }

    fn prepare_inputs(&self, prompt: &str) -> Vec<u32> {
        let encoding = self.tokenizer.encode(prompt, true).unwrap();

        let ids = encoding.get_ids();

        println!("{}", ids.len());


        ids.to_vec()
    }

    // fn run(&self, input: &str) -> Result<String> {
    //     let (tokens_tensor, pos_tensor) = self.prepare_inputs(input)?;
    //     let kv_tensor = self.dummy_kv_tensor();

    //     unsafe {
    //         wasi_nn::set_input(self.ctx, 0, tokens_tensor)?;
    //         wasi_nn::set_input(self.ctx, 1, pos_tensor)?;
    //         wasi_nn::set_input(self.ctx, 2, kv_tensor)?;
    //         wasi_nn::compute(self.ctx)?;
    //     }

    //     let output_tokens = self.get_output(0, self.max_length)?;
    //     Ok(self.decode_output(&output_tokens))
    // }

    // fn get_output(&self, index: u32, size: usize) -> Result<Vec<i32>> {
    //     let mut buffer = vec![0i32; size];
    //     let ptr = cast_slice_mut(&mut buffer).as_mut_ptr();
    //     let byte_len = (size * std::mem::size_of::<i32>()) as u32;
    //     unsafe {
    //         wasi_nn::get_output(self.ctx, index, ptr, byte_len)?;
    //     }
    //     Ok(buffer)
    // }

    // fn decode_output(&self, tokens: &[i32]) -> String {
    //     let filtered: Vec<u32> = tokens.iter().filter_map(|&id| {
    //         if id > 0 { Some(id as u32) } else { None }
    //     }).collect();
    //     self.tokenizer
    //         .decode(filtered, true)
    //         .unwrap_or_else(|_| "<DECODE ERROR>".into())
    // }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::Subscriber::builder()
        .with_max_level(tracing::Level::DEBUG)
        .with_env_filter("debug")
        .init();

    let tokenizer_path = "./models/onnx/llama3-8b-int8/tokenizer.json";
    let max_length = 512;
    let chatbot = LlamaChatbot::new(tokenizer_path, max_length)?;

    let engine = Arc::new(Engine::new(&Config::new()).unwrap());
    let module = Arc::new(Component::from_file(&engine, Path::new("../inferencer/target/wasm32-wasip1/release/inferencer.wasm")).unwrap());
    let mut instance = WasmInstance::new(engine, module, "llama3-8b-int8")?;

    println!("🦙 Chatbot ready. Type a message or 'exit':");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let input = line?;
        if input.trim().to_lowercase() == "exit" {
            break;
        }
        // prepare the tensor
        let ids = chatbot.prepare_inputs(&input);

        // infer
        let result = instance.infer_llm(ids)?;
        let response = "awaiting!";
        writeln!(stdout, "Bot: {}\n", response)?;
        stdout.flush()?;
    }

    println!("👋 Goodbye.");
    Ok(())
}


