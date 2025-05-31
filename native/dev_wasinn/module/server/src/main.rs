pub fn main() -> anyhow::Result<()>
{
    tracing_subscriber::fmt::Subscriber::builder()
        .with_max_level(tracing::Level::DEBUG)
        .with_env_filter("debug")
        .init();
    let runtime = server::create_runtime()?;
    runtime.block_on(server::start_server(3000, "../inferencer/target/wasm32-wasip1/release/inferencer.wasm".into()))
}
