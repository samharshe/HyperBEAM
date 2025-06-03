use std::{collections::HashMap, path::Path};

use anyhow::bail;
use wasmtime_wasi_nn::{backend::BackendFromDir, wit::ExecutionTarget, Graph, GraphRegistry};
pub struct Registry(HashMap<String, Graph>);
impl Registry
{
    pub fn new() -> Self
    {
        Self(HashMap::new())
    }

    pub fn load(&mut self, backend: &mut dyn BackendFromDir, path: &Path, registry_id: &str) -> anyhow::Result<()>
    {
        if !path.is_dir() {
            bail!("preload directory is not a valid directory: {}", path.display());
        }
        let graph = backend.load_from_dir(path, ExecutionTarget::Gpu)?;
        self.0.insert(registry_id.to_owned(), graph);
        Ok(())
    }
}

impl GraphRegistry for Registry
{
    fn get(&self, name: &str) -> Option<&Graph>
    {
        self.0.get(name)
    }
    fn get_mut(&mut self, name: &str) -> Option<&mut Graph>
    {
        self.0.get_mut(name)
    }
}
