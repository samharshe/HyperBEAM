use std::collections::HashMap;
use wasmtime::component::ResourceTable;
mod generated_
{
    wasmtime::component::bindgen!({
        world: "ml",
        path: "../inferencer/wit/ncl-ml.wit",
    });
}

pub mod types
{
    pub use super::generated_::{exports::ncl::ml::chatbot::SessionConfig, ncl::ml::token_generator, Ml as NclML};
}

pub struct NclMlView<'a>
{
    context: &'a mut crate::runtime::Context,
}

impl<'a> NclMlView<'a>
{
    pub fn new(context: &'a mut crate::runtime::Context) -> Self
    {
        Self {
            context,
        }
    }
}

pub struct NclMlContenx
{
    // Empty placeholder struct
}

impl Default for NclMlContenx
{
    fn default() -> Self
    {
        Self {}
    }
}
impl types::token_generator::Host for NclMlView<'_>
{
    fn generate(&mut self, session_id: types::token_generator::SessionId, token: types::token_generator::TokenId)
        -> u32
    {
        // Decode token to text and stream immediately
        let token_text = self.context.tokenizer.decode(&[token], false).unwrap_or_else(|_| format!("[token_{}]", token));
        
        // Stream token directly to broadcast channel
        match self.context.broadcast_sender.send(format!("[TEXT_TOKEN]{}", token_text)) {
            Ok(_) => 1,  // Success
            Err(_) => 0, // Channel closed or no receivers
        }
    }
}

pub fn add_to_linker<T>(
    l: &mut wasmtime::component::Linker<T>,
    f: impl Fn(&mut T) -> NclMlView<'_> + Send + Sync + Copy + 'static,
) -> anyhow::Result<()>
{
    types::token_generator::add_to_linker_get_host(l, f);
    Ok(())
}
