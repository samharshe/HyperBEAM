mod ncl_ml;
pub mod registry;
pub mod runtime;
pub mod tensor;
pub mod utils;

use std::{net::SocketAddr, path::Path, sync::Arc};

use base64;

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
use runtime::WasmInstance;
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
use utils::{full, BoxBody, InferenceRequest, TextRequest, UnifiedRequest, Result, ApiRequest, ApiRequestData, ApiResponse, ApiResponseData, ApiError};
use wasmtime::{component::Component, Config, Engine};

static NOT_FOUND: &[u8] = b"Not Found\n";
static MISSING_CONTENT_TYPE: &[u8] = b"Missing Content-Type.\n";
static JPEG_EXPECTED: &[u8] = b"Expected image/jpeg.\n";
static INTERNAL_SERVER_ERROR: &[u8] = b"Internal Server Error\n";

async fn infer(
    request: Request<Body>,
    inference_thread_sender: UnboundedSender<UnifiedRequest>,
    log_sender: tokio::sync::broadcast::Sender<String>,
) -> Result<Response<BoxBody>>
{
    log_sender.send("[server/main.rs] Received inference request.".to_string()).ok();

    // Parse JSON body
    let mut body = request.collect().await?.aggregate();
    let body_bytes = body.copy_to_bytes(body.remaining());
    
    let api_request: ApiRequest = match serde_json::from_slice(&body_bytes) {
        Ok(req) => req,
        Err(e) => {
            log_sender.send(format!("[server/main.rs] Invalid JSON request: {}", e)).ok();
            let error = ApiError {
                error: "invalid_request".to_string(),
                message: format!("Invalid JSON: {}", e),
            };
            return Ok(Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .header(header::CONTENT_TYPE, "application/json")
                .body(full(serde_json::to_string(&error).unwrap()))
                .unwrap());
        }
    };

    log_sender.send(format!("[server/main.rs] Processing request for model: {}", api_request.model)).ok();

    // Convert API request to internal request and send to inference thread
    let (sender, receiver) = oneshot::channel();
    let internal_request = match api_request.data {
        ApiRequestData::Text { prompt } => {
            log_sender.send(format!("[server/main.rs] Processing text inference: {}", prompt)).ok();
            UnifiedRequest::Text(TextRequest {
                model: api_request.model.clone(),
                prompt,
                responder: sender,
            })
        },
        ApiRequestData::Image { image } => {
            log_sender.send("[server/main.rs] Processing image inference.".to_string()).ok();
            
            // Decode base64 image
            let image_bytes = match base64::decode(&image) {
                Ok(bytes) => bytes,
                Err(e) => {
                    log_sender.send(format!("[server/main.rs] Invalid base64 image: {}", e)).ok();
                    let error = ApiError {
                        error: "invalid_image".to_string(),
                        message: format!("Invalid base64 encoding: {}", e),
                    };
                    return Ok(Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(full(serde_json::to_string(&error).unwrap()))
                        .unwrap());
                }
            };

            let tensor = tensor::jpeg_to_squeezenet_tensor(image_bytes);
            UnifiedRequest::Image(InferenceRequest {
                model: api_request.model.clone(),
                tensor_bytes: tensor,
                responder: sender,
            })
        }
    };

    // Send request to inference thread
    inference_thread_sender.send(internal_request)?;
    log_sender.send("[server/main.rs] Passed request to inferencer. Waiting for result.".to_string()).ok();

    // Wait for response
    match receiver.await {
        Ok(api_response) => {
            let json = serde_json::to_string(&api_response).unwrap();
            log_sender.send("[server/main.rs] Inference successful.".to_string()).ok();
            
            Ok(Response::builder()
                .header(header::CONTENT_TYPE, "application/json")
                .body(full(json))
                .unwrap())
        },
        Err(_) => {
            log_sender.send("[server/main.rs] Inference task failed or channel closed.".to_string()).ok();
            let error = ApiError {
                error: "inference_failed".to_string(),
                message: "Inference failed".to_string(),
            };
            Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header(header::CONTENT_TYPE, "application/json")
                .body(full(serde_json::to_string(&error).unwrap()))
                .unwrap())
        },
    }
}

async fn logs(log_sender: tokio::sync::broadcast::Sender<String>) -> Result<Response<BoxBody>>
{
    let rx = log_sender.subscribe();

    let stream = BroadcastStream::new(rx).filter_map(|msg| {
        use futures::future::ready;

        match msg {
            Ok(data) => ready(Some(Ok(Frame::data(bytes::Bytes::from(format!("data: {}\n\n", data)))))),
            Err(e) => {
                eprintln!("SSE stream error: {}", e);
                ready(None)
            },
        }
    });

    let body = StreamBody::new(stream);

    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(BoxBody::new(body))?;

    Ok(response)
}

async fn serve(
    request: Request<Body>,
    inference_thread_sender: UnboundedSender<UnifiedRequest>,
    log_sender: tokio::sync::broadcast::Sender<String>,
) -> Result<Response<BoxBody>>
{
    if request.method() == Method::OPTIONS {
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
            .header(header::ACCESS_CONTROL_ALLOW_METHODS, "*")
            .header(header::ACCESS_CONTROL_ALLOW_HEADERS, "*")
            .body(full(""))
            .unwrap());
    }

    let mut response = match (request.method(), request.uri().path()) {
        (&Method::GET, "/logs") => logs(log_sender.clone()).await?,
        (&Method::POST, "/infer") => infer(request, inference_thread_sender, log_sender.clone()).await?,
        _ => {
            log_sender
                .send(format!("[server/main.rs] Unhandled request: {} {}", request.method(), request.uri().path()))
                .ok();
            Response::builder().status(StatusCode::NOT_FOUND).body(full(NOT_FOUND)).unwrap()
        },
    };

    let headers = response.headers_mut();
    headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*".parse().unwrap());
    headers.insert(header::ACCESS_CONTROL_ALLOW_METHODS, "*".parse().unwrap());
    headers.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, "*".parse().unwrap());

    Ok(response)
}

pub fn create_runtime() -> std::io::Result<Runtime>
{
    Builder::new_multi_thread().enable_all().worker_threads(8).build()
}

pub async fn start_server(port: u16, wasm_module_path: String) -> anyhow::Result<()>
{
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = TcpListener::bind(addr).await?;
    println!("✅ HTTP server listening on http://127.0.0.1:{}", port);
    let (tx, mut rx) = unbounded_channel::<UnifiedRequest>();
    let (log_tx, _log_rx) = tokio::sync::broadcast::channel::<String>(16);
    let log_tx_inference = log_tx.clone();
    tokio::spawn(async move {
        log_tx_inference.send("Inference thread is active and working.".to_string()).ok();
        let engine = Arc::new(Engine::new(&Config::new()).unwrap());
        let module = Arc::new(Component::from_file(&engine, Path::new(&wasm_module_path)).unwrap());
        
        // Load tokenizer for text inference
        let tokenizer_path = "./models/onnx/llama3.1-8b-instruct/tokenizer.json";
        let tokenizer = match tokenizers::tokenizer::Tokenizer::from_file(tokenizer_path) {
            Ok(t) => Some(t),
            Err(e) => {
                log_tx_inference.send(format!("Failed to load tokenizer: {}", e)).ok();
                None
            }
        };
        
        while let Some(request) = rx.recv().await {
            let engine = Arc::clone(&engine);
            let module = Arc::clone(&module);
            let tokenizer = tokenizer.clone();
            let log_tx_clone = log_tx_inference.clone();
            
            spawn_blocking(move || -> anyhow::Result<()> {
                match request {
                    UnifiedRequest::Image(image_req) => {
                        // Handle image inference (currently incomplete)
                        log_tx_clone.send("Processing image inference...".to_string()).ok();
                        let instance = WasmInstance::new(engine, module, &image_req.model)?;
                        // TODO: Implement image inference
                        // let result = instance.infer(image_req.tensor_bytes)?;
                        let api_response = ApiResponse {
                            model: image_req.model,
                            result: ApiResponseData::Image { 
                                label: 0, // TODO: Actual inference
                                probability: 0.0 
                            },
                        };
                        let _ = image_req.responder.send(api_response);
                        Ok(())
                    },
                    UnifiedRequest::Text(text_req) => {
                        // Handle text inference
                        log_tx_clone.send(format!("Processing text inference for: {}", text_req.prompt)).ok();
                        
                        if let Some(ref tokenizer) = tokenizer {
                            let mut instance = WasmInstance::new(engine, module, &text_req.model)?;
                            match instance.infer_text(&text_req.prompt, tokenizer) {
                                Ok(response_text) => {
                                    log_tx_clone.send(format!("Text inference completed: {}", response_text)).ok();
                                    let api_response = ApiResponse {
                                        model: text_req.model,
                                        result: ApiResponseData::Text { text: response_text },
                                    };
                                    let _ = text_req.responder.send(api_response);
                                },
                                Err(e) => {
                                    log_tx_clone.send(format!("Text inference failed: {}", e)).ok();
                                    let api_response = ApiResponse {
                                        model: text_req.model,
                                        result: ApiResponseData::Text { text: format!("Error: {}", e) },
                                    };
                                    let _ = text_req.responder.send(api_response);
                                }
                            }
                        } else {
                            log_tx_clone.send("Tokenizer not available for text inference".to_string()).ok();
                            let api_response = ApiResponse {
                                model: text_req.model,
                                result: ApiResponseData::Text { text: "Error: Tokenizer not available".to_string() },
                            };
                            let _ = text_req.responder.send(api_response);
                        }
                        Ok(())
                    }
                }
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
