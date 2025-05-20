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

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct InferenceResult(pub u32, pub f32);

#[derive(Debug)]
pub struct InferenceRequest
{
    pub tensor_bytes: Vec<u8>,
    pub responder: oneshot::Sender<InferenceResult>,
}
