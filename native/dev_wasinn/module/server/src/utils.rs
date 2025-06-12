use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

type GenericError = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, GenericError>;
pub type BoxBody = http_body_util::combinators::BoxBody<Bytes, hyper::Error>;

pub fn full<T: Into<Bytes>>(chunk: T) -> BoxBody
{
    Full::new(chunk.into()).map_err(|never| match never {}).boxed()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextGenerationParams {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

fn default_max_tokens() -> u32 { 50 }

impl Default for TextGenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct InferenceResult(pub u32, pub f32);

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct TextResult
{
    pub text: String,
}

#[derive(Debug)]
pub struct InferenceRequest
{
    pub model: String,
    pub tensor_bytes: Vec<u8>,
    pub responder: oneshot::Sender<ApiResponseData>,
}

#[derive(Debug)]
pub struct TextRequest
{
    pub model: String,
    pub prompt: String,
    pub params: TextGenerationParams,
    pub responder: oneshot::Sender<ApiResponseData>,
}

#[derive(Debug)]
pub enum UnifiedRequest
{
    Image(InferenceRequest),
    Text(TextRequest),
}

// JSON API Request/Response structures
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiRequest {
    pub model: String,
    #[serde(flatten)]
    pub data: ApiRequestData,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ApiRequestData {
    Text { 
        prompt: String,
        #[serde(flatten)]
        params: TextGenerationParams,
    },
    Image { image: String }, // Base64 encoded image
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ApiResponseData {
    Text { text: String },
    Image { label: u32, probability: f32 },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiError {
    pub error: String,
    pub message: String,
}
