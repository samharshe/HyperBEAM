use server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🚀 Starting server and loading module...");
    
    let wasm_module_path = "../inferencer/target/wasm32-wasip1/release/ncl_ml.wasm".to_string();
    
    server::start_server(3000, wasm_module_path).await
}