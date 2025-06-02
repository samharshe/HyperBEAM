use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
use wasmtime::{
    component::{Component, Instance, Linker, ResourceTable},
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

use super::{registry::Registry, utils::InferenceResult};

struct Context
{
    wasi: WasiCtx,
    wasi_nn: WasiNnCtx,
    table: ResourceTable,
}

impl Context
{
    fn new(preopen_dir: &Path, preload_model: bool, mut backend: Backend, registry_id: &str) -> Result<Self>
    {
        let mut builder = WasiCtxBuilder::new();
        builder.inherit_stdio().preopened_dir(preopen_dir, "", DirPerms::READ, FilePerms::READ)?;
        let wasi = builder.build();

        let mut registry = Registry::new();
        if preload_model {
            registry.load((backend).as_dir_loadable().unwrap(), preopen_dir, registry_id)?;
        }
        let wasi_nn = WasiNnCtx::new([backend], registry.into());
        let table = ResourceTable::new();
        Ok(Self {
            wasi,
            wasi_nn,
            table,
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
    instance: Instance,
    store: Store<Context>,
    registry_id: String, // memory: Memory,
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
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;
        let instance: Instance = linker.instantiate(&mut store, &component)?;
        Ok(Self {
            instance,
            store,
            registry_id: registry_id.to_owned(),
        })
    }

    pub fn infer(&mut self, tensor_bytes: Vec<u8>) -> anyhow::Result<InferenceResult>
    {
        let interface_idx = self
            .instance
            .get_export_index(&mut self.store, None, "component:inferer/mobilenet@0.1.0")
            .expect("Cannot get `component:inferer/mobilenet@0.1.0` interface");

        let parent_export_idx = Some(&interface_idx);
        let func_idx = self
            .instance
            .get_export_index(&mut self.store, parent_export_idx, "infer")
            .expect("Cannot find `infer` function in `component:inferer/mobilenet@0.1.0` interface");
        let func = self.instance.get_func(&mut self.store, func_idx).expect("func_idx is unexpectedly missing");
        let infer = func.typed::<(String, Vec<u8>), ((u32, f32),)>(&self.store)?;
        let ((label, confidence),) = infer.call(&mut self.store, (self.registry_id.clone(), tensor_bytes))?;
        Ok(InferenceResult(label, confidence))
    }

    pub fn infer_llm(&mut self, ids: Vec<i64>) -> anyhow::Result<()> {
        let interface_idx = self
            .instance
            .get_export_index(&mut self.store, None, "component:inferer/mobilenet@0.1.0")
            .expect("Cannot get `component:inferer/mobilenet@0.1.0` interface");

        let parent_export_idx = Some(&interface_idx);
        let func_idx = self
            .instance
            .get_export_index(&mut self.store, parent_export_idx, "infer-llm")
            .expect("Cannot find `infer` function in `component:inferer/mobilenet@0.1.0` interface");
        let func = self.instance.get_func(&mut self.store, func_idx).expect("func_idx is unexpectedly missing");
        let infer = func.typed::<(String, Vec<i64>), ()>(&self.store)?;
        infer.call(&mut self.store, (self.registry_id.clone(), ids,))?;
        Ok(())
    }
}
