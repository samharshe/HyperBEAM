use std::{path::Path, sync::Arc};

use anyhow::Result;
use wasmtime::{Engine as WasmEngine, Instance, Linker, Memory, Module, Store};
use wasmtime_wasi::{preview1::WasiP1Ctx, p2::WasiCtxBuilder, DirPerms, FilePerms};
use wasmtime_wasi_nn::{backend::openvino::OpenvinoBackend, witx::WasiNnCtx, Backend, InMemoryRegistry};

use super::utils::InferenceResult;

struct Context
{
    wasi: WasiP1Ctx,
    wasi_nn: WasiNnCtx,
}

impl Context
{
    fn new(preopen_dir: &Path, preload_model: bool, mut backend: Backend) -> Result<Self>
    {
        let mut builder = WasiCtxBuilder::new();
        builder.inherit_stdio().preopened_dir(preopen_dir, "fixture", DirPerms::READ, FilePerms::READ)?;
        let wasi = builder.build_p1();

        // For OpenVINO backend, we'll let the WASM module load the model  
        let registry = InMemoryRegistry::new();
        let wasi_nn = WasiNnCtx::new([backend], registry.into());

        Ok(Self {
            wasi,
            wasi_nn,
        })
    }
}

pub struct WasmInstance
{
    instance: Instance,
    store: Store<Context>,
    memory: Memory,
}

impl WasmInstance
{
    pub fn new(engine: Arc<WasmEngine>, module: Arc<Module>) -> anyhow::Result<WasmInstance>
    {
        let path = Path::new("./fixture");
        let mut store =
            Store::new(&engine, Context::new(path, true, Backend::from(OpenvinoBackend::default())).unwrap());

        let mut linker = Linker::new(&engine);
        wasmtime_wasi_nn::witx::add_to_linker(&mut linker, |s: &mut Context| &mut s.wasi_nn)?;
        wasmtime_wasi::preview1::add_to_linker_sync(&mut linker, |s: &mut Context| &mut s.wasi)?;

        let instance = linker.instantiate(&mut store, &module)?;
        let memory = instance.get_memory(&mut store, "memory").expect("failed to find 'memory' export");

        Ok(Self {
            instance,
            store,
            memory,
        })
    }

    pub fn infer(&mut self, tensor_bytes: Vec<u8>) -> anyhow::Result<InferenceResult>
    {
        let result_ptr = 1000; // 8 bytes result
        let ptr = 1008;
        self.memory.write(&mut self.store, ptr, &tensor_bytes)?;

        let infer = self.instance.get_typed_func::<(i32, i32, i32), i32>(&mut self.store, "infer")?;

        let status = infer.call(&mut self.store, (ptr as i32, tensor_bytes.len() as i32, result_ptr))?;
        if status != 0 {
            return Err(anyhow::anyhow!("Inference failed with status: {}", status));
        }
        let mut result_buf = [0u8; 8];
        self.memory.read(&mut self.store, result_ptr as usize, &mut result_buf)?;
        let label_index = u32::from_le_bytes(result_buf[0..4].try_into().unwrap());
        let confidence = f32::from_le_bytes(result_buf[4..8].try_into().unwrap());

        Ok(InferenceResult(label_index, confidence))
    }
}