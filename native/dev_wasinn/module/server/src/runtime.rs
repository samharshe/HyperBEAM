use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use tokio::sync::mpsc::UnboundedReceiver;
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

struct Context
{
    wasi: WasiCtx,
    wasi_nn: WasiNnCtx,
    ncl_ml: ncl_ml::NclMlContenx,
    table: ResourceTable,
}

impl Context
{
    fn new(preopen_dir: &Path, preload_model: bool, mut backend: Backend, registry_id: &str) -> anyhow::Result<Self>
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
    pub fn new(engine: Arc<Engine>, component: Arc<Component>, registry_id: &str) -> anyhow::Result<WasmInstance>
    {
        let full_path: PathBuf = std::env::current_dir().unwrap().join("models").join("onnx").join(registry_id);

        let mut store = Store::new(
            &engine,
            Context::new(&full_path, true, Backend::from(OnnxBackend::default()), registry_id).unwrap(),
        );

        let mut linker = Linker::new(&engine);
        wasmtime_wasi_nn::wit::add_to_linker(&mut linker, |c: &mut Context| {
            WasiNnView::new(&mut c.table, &mut c.wasi_nn)
        })?;
        ncl_ml::add_to_linker(&mut linker, |c: &mut Context| {
            ncl_ml::NclMlView::new(&mut c.table, &mut c.ncl_ml)
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

    pub fn register(&mut self) -> anyhow::Result<(u64, UnboundedReceiver<u32>)>
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
        
        // Create session in the ncl_ml context to receive tokens
        let rx = self.store.data_mut().ncl_ml.new_session(session_id);
        
        Ok((session_id, rx))
    }
    pub fn infer_llm(&mut self, session_id: u64, ids: Vec<i64>, token_receiver: &mut UnboundedReceiver<u32>) -> anyhow::Result<Vec<u32>>
    {
        // Call the WASM component's chatbot::infer method
        match self.ncl_ml_world.ncl_ml_chatbot().call_infer(&mut self.store, session_id, &ids) {
            Ok(result) => match result {
                Ok(_) => {
                    // Collect all tokens generated during inference
                    let mut tokens = Vec::new();
                    
                    // Drain the entire channel
                    loop {
                        match token_receiver.try_recv() {
                            Ok(token) => tokens.push(token),
                            Err(_) => break, // Channel empty
                        }
                    }
                    
                    if tokens.is_empty() {
                        eprintln!("No tokens received from WASM component");
                    } else {
                        eprintln!("Collected {} tokens from WASM component: {:?}", tokens.len(), tokens);
                    }
                    
                    Ok(tokens)
                },
                Err(error) => {
                    eprintln!("WASM component inference error: {:?}", error);
                    Ok(vec![])
                }
            },
            Err(error) => {
                eprintln!("WASM component call error: {}", error);
                Ok(vec![])
            }
        }
    }

    /// Complete text inference from prompt to response text
    pub fn infer_text(&mut self, user_prompt: &str, tokenizer: &tokenizers::tokenizer::Tokenizer) -> anyhow::Result<String>
    {
        // Format the prompt with system message template
        let prompt = format!(r#"<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"#, user_prompt);
        
        // Tokenize the prompt
        let encoding = tokenizer.encode(prompt.clone(), false).unwrap();
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        
        eprintln!("DEBUG runtime: Formatted prompt: {}", prompt);
        eprintln!("DEBUG runtime: Tokenized to {} tokens", ids.len());
        
        // Register a session if we don't have one, or reuse existing
        let (session_id, mut session_receiver) = self.register()?;
        
        // Run inference
        let generated_tokens = self.infer_llm(session_id, ids, &mut session_receiver)?;
        
        if generated_tokens.is_empty() {
            return Ok("".to_string());
        }
        
        // Decode tokens back to text
        let response = tokenizer.decode(&generated_tokens, false).unwrap();
        
        eprintln!("DEBUG runtime: Generated response: {}", response);
        Ok(response)
    }
}
