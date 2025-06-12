#!/bin/bash
set -e

echo "Building dev_wasinn_nif rust project..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# build the inferencer wasm module
cargo component build --manifest-path="$SCRIPT_DIR/inferencer/Cargo.toml"
cp "$SCRIPT_DIR/inferencer/target/wasm32-wasip1/release/inferencer.wasm" "$SCRIPT_DIR/../../priv/"

# build the server and the nif glue
cargo build --manifest-path="$SCRIPT_DIR/Cargo.toml" --release

cp "$SCRIPT_DIR/target/release/libdev_wasinn_nif.so" "$SCRIPT_DIR/../../priv/"

cp -R "$SCRIPT_DIR/server/fixture" "$SCRIPT_DIR/../../priv/"
