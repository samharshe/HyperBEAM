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
        types::{NclML, SessionConfig},
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
        let guest = self.ncl_ml_world.ncl_ml_chatbot();
        let result = guest.call_register(
            &mut self.store,
            &SessionConfig {
                model_id: self.registry_id.clone(),
                max_token: Some(10),
                history: None,
            },
        )?;
        let ctx = self.store.data_mut();
        Ok((result, ctx.ncl_ml.new_session(result)))
    }
    pub fn infer_llm(&mut self, session_id: u64, ids: Vec<i64>) -> anyhow::Result<Vec<u32>>
    {
        let guest = self.ncl_ml_world.ncl_ml_chatbot();
        let result = guest.call_infer(&mut self.store, session_id, &ids)?;
        Ok(vec![])
    }
}
