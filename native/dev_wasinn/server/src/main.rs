pub fn main() -> anyhow::Result<()>
{
    let runtime = server::create_runtime()?;
    runtime.block_on(server::start_server(3000, "../target/wasm32-wasip1/debug/inferencer.wasm".into()))
}
