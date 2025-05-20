#!/bin/bash
set -e

echo "Cleaning dev_wasinn_nif rust build..."


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cargo clean --manifest-path="$SCRIPT_DIR/Cargo.toml"

rm -f "$SCRIPT_DIR/../../priv/libdev_wasinn_nif.so"
