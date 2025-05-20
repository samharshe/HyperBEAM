define USAGE
REQUIREMENTS:
- Cargo

USAGE:
> make [
	sense: see usage.
	clean: remove the build environment.
	serve: run in dev move.
	build: build the production executable.
]
endef

PWD = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

clean:
	@cargo clean

inferencer-build: 
	cargo build --package inferencer --target wasm32-wasip1 --release

server-build:
	cargo build --package server --release

format:
	@cargo fix
	@cargo +nightly fmt

lint:
	@cargo clippy --all-targets --all-features -- -D warnings