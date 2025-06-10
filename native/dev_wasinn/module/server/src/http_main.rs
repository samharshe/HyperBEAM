use server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🌐 Starting HTTP server on port 3000...");
    println!("📁 Loading WASM module...");
    
    let wasm_module_path = "../inferencer/target/wasm32-wasip1/release/ncl_ml.wasm".to_string();
    
    println!("🚀 Server starting (this may take a moment to load models)...");
    server::start_server(3000, wasm_module_path).await
}