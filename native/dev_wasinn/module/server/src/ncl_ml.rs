use std::collections::HashMap;

use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
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
    ctx: &'a mut NclMlContenx,
    table: &'a mut ResourceTable,
}

impl<'a> NclMlView<'a>
{
    pub fn new(table: &'a mut ResourceTable, ctx: &'a mut NclMlContenx) -> Self
    {
        Self {
            ctx,
            table,
        }
    }
}

pub struct NclMlContenx
{
    sessions: HashMap<u64, UnboundedSender<u32>>,
}

impl Default for NclMlContenx
{
    fn default() -> Self
    {
        Self {
            sessions: HashMap::new(),
        }
    }
}

impl NclMlContenx
{
    pub fn new_session(&mut self, session_id: u64) -> UnboundedReceiver<u32>
    {
        let (sender, receiver) = unbounded_channel::<u32>();
        self.sessions.insert(session_id, sender);
        receiver
    }
}
impl types::token_generator::Host for NclMlView<'_>
{
    fn generate(&mut self, session_id: types::token_generator::SessionId, token: types::token_generator::TokenId)
        -> u32
    {
        match self.ctx.sessions.get(&session_id) {
            None => 0,
            Some(session) => {
                // Send the token through the channel
                match session.send(token) {
                    Ok(_) => 1,  // Success
                    Err(_) => 0, // Channel closed
                }
            },
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
