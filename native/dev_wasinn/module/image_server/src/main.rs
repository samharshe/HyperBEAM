mod runtime;
mod tensor;
mod utils;

use std::{net::SocketAddr, path::Path, sync::Arc};

use bytes::Buf;
use http_body_util::BodyExt;
use hyper::{
    body::{
        Incoming as Body,
        Frame
    },
    header, server::conn::http1,
    service::service_fn, Method,
    Request,
    Response,
    StatusCode
};
use hyper_util::rt::{TokioIo, TokioTimer};
use runtime::WasmInstance;
use tokio::{
    net::TcpListener,
    sync::{
        mpsc::{unbounded_channel, UnboundedSender},
        oneshot,
        broadcast,
    },
    task::spawn_blocking,
};
use utils::{full, BoxBody, InferenceRequest, Result};
use wasmtime::{Config, Engine, Module};
use futures::stream::{StreamExt};
use http_body_util::StreamBody;
use tokio_stream::wrappers::BroadcastStream;

static NOT_FOUND: &[u8] = b"Not Found\n";
static MISSING_CONTENT_TYPE: &[u8] = b"Missing Content-Type.\n";
static JPEG_EXPECTED: &[u8] = b"Expected image/jpeg.\n";
static INTERNAL_SERVER_ERROR: &[u8] = b"Internal Server Error\n";

async fn infer(
    request: Request<Body>,
    inference_thread_sender: UnboundedSender<InferenceRequest>,
    log_sender: tokio::sync::broadcast::Sender<String>,
) -> Result<Response<BoxBody>>
{
    log_sender.send("[server/main.rs] Received inference request.".to_string()).ok();

    if let Some(content_type) = request.headers().get(header::CONTENT_TYPE) {
        if content_type != "image/jpeg" {
            log_sender.send("[server/main.rs] Inference request has unsupported media type.".to_string()).ok();
            return Ok(Response::builder()
                .status(StatusCode::UNSUPPORTED_MEDIA_TYPE)
                .body(full(JPEG_EXPECTED))
                .unwrap());
        }
        let mut body = request.collect().await?.aggregate();
        let bytes = body.copy_to_bytes(body.remaining());

        log_sender.send("[server/main.rs] Processing image for inference.".to_string()).ok();
        let tensor = tensor::jpeg_to_raw_bgr(bytes.to_vec(), &log_sender).unwrap();
        let (sender, receiver) = oneshot::channel();
        inference_thread_sender.send(InferenceRequest {
            tensor_bytes: tensor,
            responder: sender,
        })?;

        log_sender.send("[server/main.rs] Passed tensor to inferencer. Waiting for inference result.".to_string()).ok();
        log_sender.send("[server/main.rs] (No logs during inference because this is performed by WASM module.)".to_string()).ok();
        return match receiver.await {
            Ok(response) => {
                let r = match serde_json::to_string(&response) {
                    Ok(json) => {
                        log_sender.send(format!("[server/main.rs] Inference result: [label={}, probability={:.5}].", response.0, response.1)).ok();
                        log_sender.send("[server/main.rs] Inference successful. Sending response.".to_string()).ok();
                        Response::builder().header(header::CONTENT_TYPE, "application/json").body(full(json)).unwrap()
                    },
                    Err(_) => {
                        log_sender.send("[server/main.rs] Error serializing inference response.".to_string()).ok();
                        Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(full(INTERNAL_SERVER_ERROR))
                        .unwrap()
                    },
                };
                Ok(r)
            },
            Err(_) => {
                log_sender.send("[server/main.rs] Inference task failed or channel closed.".to_string()).ok();
                Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(full(INTERNAL_SERVER_ERROR))
                .unwrap())
            },
        };
    }

    log_sender.send("[server/main.rs] Inference request missing Content-Type header.".to_string()).ok();
    Ok(Response::builder().status(StatusCode::BAD_REQUEST).body(full(MISSING_CONTENT_TYPE)).unwrap())
}

async fn logs(
    log_sender: tokio::sync::broadcast::Sender<String>,
) -> Result<Response<BoxBody>> {
    let rx = log_sender.subscribe();

    let stream = BroadcastStream::new(rx)
        .filter_map(|msg| {
            use futures::future::ready;

            match msg {
                Ok(data) => {
                    ready(Some(Ok(Frame::data(bytes::Bytes::from(format!("data: {}\n\n", data))))))
                },
                Err(e) => {
                    eprintln!("SSE stream error: {}", e);
                    ready(None)
                }
            }
        });

    let body = StreamBody::new(stream);

    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "http://localhost:8000")
        .body(BoxBody::new(body))?;

    Ok(response)
}

async fn serve(
    request: Request<Body>,
    inference_thread_sender: UnboundedSender<InferenceRequest>,
    log_sender: tokio::sync::broadcast::Sender<String>,
) -> Result<Response<BoxBody>> {
    if request.method() == Method::OPTIONS {
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "http://localhost:8000")
            .header(header::ACCESS_CONTROL_ALLOW_METHODS, "POST, OPTIONS, GET")
            .header(header::ACCESS_CONTROL_ALLOW_HEADERS, "Content-Type, x-session-id")
            .body(full(""))
            .unwrap());
    }

    let mut response = match (request.method(), request.uri().path()) {
        (&Method::GET, "/logs") => logs(log_sender.clone()).await?,
        (&Method::POST, "/infer") => infer(request, inference_thread_sender, log_sender.clone()).await?,
        _ => {
            log_sender.send(format!("[server/main.rs] Unhandled request: {} {}", request.method(), request.uri().path())).ok();
            Response::builder().status(StatusCode::NOT_FOUND).body(full(NOT_FOUND)).unwrap()
        },
    };

    let headers = response.headers_mut();
    headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, "http://localhost:8000".parse().unwrap());
    headers.insert(header::ACCESS_CONTROL_ALLOW_METHODS, "POST, OPTIONS, GET".parse().unwrap());
    headers.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, "Content-Type, x-session-id".parse().unwrap());

    Ok(response)
}

#[tokio::main]
pub async fn main() -> anyhow::Result<()>
{
    let addr: SocketAddr = ([127, 0, 0, 1], 3000).into();
    let listener = TcpListener::bind(addr).await?;
    let (tx, mut rx) = unbounded_channel::<InferenceRequest>();

    let (log_tx, _log_rx) = broadcast::channel::<String>(16);
    let log_tx_inference = log_tx.clone();

    tokio::spawn(async move {
        log_tx_inference.send("Inference thread is active and working.".to_string()).ok();
        let engine = Arc::new(Engine::new(&Config::new()).unwrap());
        let module =
            Arc::new(Module::from_file(&engine, Path::new("../target/wasm32-wasip1/debug/inferencer.wasm")).unwrap());
        while let Some(request) = rx.recv().await {
            let engine = Arc::clone(&engine);
            let module = Arc::clone(&module);
            spawn_blocking(move || -> anyhow::Result<()> {
                let mut instance = WasmInstance::new(engine, module)?;
                let result = instance.infer(request.tensor_bytes).unwrap();
                request.responder.send(result).unwrap();
                Ok(())
            });
        }
    });

    loop {
        let (tcp, _) = listener.accept().await?;
        let io = TokioIo::new(tcp);
        let tx = tx.clone();
        let log_tx_clone = log_tx.clone();
        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .timer(TokioTimer::new())
                .serve_connection(io, service_fn(move |req| serve(req, tx.clone(), log_tx_clone.clone())))
                .await
            {
                println!("Error serving connection: {:?}", err);
            }
        });
    }
}