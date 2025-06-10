use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use wasmtime::{
    component::{Component, Linker, ResourceTable},
    Engine, Store,
};
use wasmtime_wasi::{
    p2::{WasiCtx, WasiCtxBuilder},
    DirPerms, FilePerms,
};
use wasmtime_wasi_nn::{
    backend::onnx::OnnxBackend,
    wit::{WasiNnCtx, WasiNnView},
    Backend,
};

use super::{
    ncl_ml::{
        self,
        types::NclML,
    },
    registry::Registry,
};

pub struct Context
{
    pub wasi: WasiCtx,
    pub wasi_nn: WasiNnCtx,
    pub ncl_ml: ncl_ml::NclMlContenx,
    pub table: ResourceTable,
    pub broadcast_sender: tokio::sync::broadcast::Sender<String>,
    pub tokenizer: tokenizers::tokenizer::Tokenizer,
}

impl Context
{
    fn new(preopen_dir: &Path, preload_model: bool, mut backend: Backend, registry_id: &str, broadcast_sender: tokio::sync::broadcast::Sender<String>, tokenizer: tokenizers::tokenizer::Tokenizer) -> anyhow::Result<Self>
    {
        let mut builder = WasiCtxBuilder::new();
        builder.inherit_stdio().preopened_dir(preopen_dir, "", DirPerms::READ, FilePerms::READ)?;
        let wasi = builder.build();

        let mut registry = Registry::new();
        if preload_model {
            registry.load((backend).as_dir_loadable().unwrap(), preopen_dir, registry_id)?;
        }
        let wasi_nn = WasiNnCtx::new([backend], registry.into());
        Ok(Self {
            wasi,
            wasi_nn,
            table: ResourceTable::new(),
            ncl_ml: ncl_ml::NclMlContenx::default(),
            broadcast_sender,
            tokenizer,
        })
    }
}
impl wasmtime_wasi::p2::IoView for Context
{
    fn table(&mut self) -> &mut ResourceTable
    {
        &mut self.table
    }
}
impl wasmtime_wasi::p2::WasiView for Context
{
    fn ctx(&mut self) -> &mut WasiCtx
    {
        &mut self.wasi
    }
}

pub struct WasmInstance
{
    ncl_ml_world: NclML,
    store: Store<Context>,
    registry_id: String,
}

impl WasmInstance
{
    pub fn new(engine: Arc<Engine>, component: Arc<Component>, registry_id: &str, broadcast_sender: tokio::sync::broadcast::Sender<String>, tokenizer: tokenizers::tokenizer::Tokenizer) -> anyhow::Result<WasmInstance>
    {
        let full_path: PathBuf = std::env::current_dir().unwrap().join("models").join("onnx").join(registry_id);

        let mut store = Store::new(
            &engine,
            Context::new(&full_path, true, Backend::from(OnnxBackend::default()), registry_id, broadcast_sender, tokenizer).unwrap(),
        );

        let mut linker = Linker::new(&engine);
        wasmtime_wasi_nn::wit::add_to_linker(&mut linker, |c: &mut Context| {
            WasiNnView::new(&mut c.table, &mut c.wasi_nn)
        })?;
        ncl_ml::add_to_linker(&mut linker, |c: &mut Context| {
            ncl_ml::NclMlView::new(c)
        })?;
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;
        let ncl_ml_world = NclML::instantiate(&mut store, &component, &linker)?;
        // let instance: Instance = linker.instantiate(&mut store, &component)?;
        Ok(Self {
            ncl_ml_world,
            store,
            registry_id: registry_id.to_owned(),
        })
    }

    pub fn register(&mut self) -> anyhow::Result<u64>
    {
        use crate::ncl_ml::types::SessionConfig;
        
        // Create session config for the model
        let config = SessionConfig {
            model_id: self.registry_id.clone(),
            history: None,
            max_token: Some(50),
        };
        
        // Register session with WASM component
        let session_id = self.ncl_ml_world.ncl_ml_chatbot().call_register(&mut self.store, &config)?;
        
        Ok(session_id)
    }
    pub fn infer_llm(&mut self, session_id: u64, ids: Vec<i64>, log_sender: Option<tokio::sync::broadcast::Sender<String>>) -> anyhow::Result<()>
    {
        // Debug: Show we entered infer_llm
        if let Some(ref sender) = log_sender {
            sender.send("[DEBUG] Inside infer_llm, about to call WASM component".to_string()).ok();
        }
        
        // Call the WASM component's chatbot::infer method
        match self.ncl_ml_world.ncl_ml_chatbot().call_infer(&mut self.store, session_id, &ids) {
            Ok(result) => {
                if let Some(ref sender) = log_sender {
                    sender.send("[DEBUG] WASM call_infer returned Ok, checking inner result".to_string()).ok();
                }
                match result {
                    Ok(_) => {
                        if let Some(ref sender) = log_sender {
                            sender.send("[DEBUG] Inference completed successfully".to_string()).ok();
                        }
                        Ok(())
                    },
                    Err(error) => {
                        if let Some(ref sender) = log_sender {
                            sender.send(format!("[DEBUG] Inner result is Err: {:?}", error)).ok();
                        }
                        eprintln!("WASM component inference error: {:?}", error);
                        Err(anyhow::anyhow!("WASM inference error: {:?}", error))
                    }
                }
            },
            Err(error) => {
                if let Some(ref sender) = log_sender {
                    sender.send(format!("[DEBUG] WASM call_infer failed: {}", error)).ok();
                }
                eprintln!("WASM component call error: {}", error);
                Err(anyhow::anyhow!("WASM call error: {}", error))
            }
        }
    }

    /// Complete text inference from prompt to response text
    pub fn infer_text(&mut self, user_prompt: &str, log_sender: Option<tokio::sync::broadcast::Sender<String>>) -> anyhow::Result<String>
    {
        // Format the prompt with system message template
        let prompt = format!(r#"<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"#, user_prompt);
        
        // Tokenize the prompt
        let encoding = self.store.data().tokenizer.encode(prompt.clone(), false).unwrap();
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        
        eprintln!("DEBUG runtime: Formatted prompt: {}", prompt);
        eprintln!("DEBUG runtime: Tokenized to {} tokens", ids.len());
        
        // Register a session if we don't have one, or reuse existing
        let session_id = self.register()?;
        
        // Run inference
        if let Some(ref sender) = log_sender {
            sender.send("[DEBUG] About to call infer_llm with streaming".to_string()).ok();
        }
        self.infer_llm(session_id, ids, log_sender)?;
        
        // Since tokens are now streaming directly to broadcast channel,
        // we don't collect them here. Return a placeholder response.
        Ok("Inference completed - tokens streamed via SSE".to_string())
    }
}
