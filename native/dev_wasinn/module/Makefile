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
	@cd inferencer && cargo clean

inferencer-build: 
	@cd inferencer && cargo component build --release

server-build:
	@cd server && cargo build --release

format:
	@cd server && cargo fix
	@cd server && cargo +nightly fmt
	@cd inferencer && cargo fix
	@cd inferencer && cargo +nightly fmt

lint:
	@cargo clippy --all-targets --all-features -- -D warnings