use std::sync::{Mutex, OnceLock};
use std::thread;

use rustler::NifResult;
use server;

static SERVER_STARTED: OnceLock<Mutex<bool>> = OnceLock::new();

#[rustler::nif(name = "load_http_server")]
fn load_http_server(port: u16, path: String) -> NifResult<String> {
    let handle_cell = SERVER_STARTED.get_or_init(|| Mutex::new(false));

    let mut lock = handle_cell.lock().unwrap();
    if *lock {
        return Ok("Server already running".into());
    }

    thread::spawn(move || {
        let runtime = server::create_runtime().expect("Tokio runtime not created!");
        runtime.block_on(server::start_server(port, path));
    });

    *lock = true;
    Ok("Server started".into())
}

rustler::init!("dev_wasinn_nif");
