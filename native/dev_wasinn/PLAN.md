# Image Classification Implementation Plan (Integrate Demo First, Then Refactor)

## Overview

This plan implements image classification using a two-phase approach: first integrate cascadia-demo's proven MobileNet implementation to achieve a working state ("green"), then refactor to Component Model architecture. This minimizes risk by establishing a working baseline before attempting the Component Model modernization.

## Strategy: Integrate First, Refactor Second

### Phase 1: Get Green (Integration)
**Goal**: Integrate cascadia-demo's working inference with current module's JSON API
**Timeline**: 2-4 hours
**Risk**: Low - copying proven code

### Phase 2: Refactor to Component Model  
**Goal**: Replace FFI interface with Component Model while preserving functionality
**Timeline**: 1-2 days
**Risk**: Medium - Component Model compatibility unknown, but with working fallback

## Architecture Evolution

### Current State (Broken)
```
Frontend (JSON) → Server (placeholder) → [Missing WASM] → No Results
```

### After Phase 1 (Green - FFI)
```
Frontend (JSON) → Server (JSON→JPEG adapter) → WASM (FFI) → OpenVINO → MobileNet → Results
```

### After Phase 2 (Green - Component Model)
```
Frontend (JSON) → Server (preprocessing) → WASM Component (WIT) → OpenVINO → MobileNet → Results
```

## Phase 1: Integration - Get to Working State

### Goal: Copy Cascadia-Demo + Create JSON API Adapter

This phase integrates cascadia-demo's proven inference pipeline with the current module's JSON API structure, requiring minimal changes to cascadia-demo's core logic.

### Step 1.1: Copy Core Infrastructure

**Copy cascadia-demo files to current module structure:**

```bash
# Copy proven inference logic
cp cascadia-demo/inferencer/src/lib.rs module/inferencer/src/lib.rs
cp cascadia-demo/inferencer/src/main.rs module/inferencer/src/main.rs  
cp cascadia-demo/inferencer/Cargo.toml module/inferencer/Cargo.toml

# Copy proven server runtime
cp cascadia-demo/server/src/runtime.rs module/server/src/runtime.rs

# Copy proven image processing  
cp cascadia-demo/server/src/tensor.rs module/server/src/tensor.rs

# Copy model files
cp -r cascadia-demo/server/fixture/ module/server/

# Copy working dependencies
cp cascadia-demo/server/Cargo.toml module/server/Cargo.toml
```

**Key Principle**: Copy exactly, modify minimally.

### Step 1.2: Create JSON API Adapter

**File**: `module/server/src/lib.rs`

**Current Broken Code** (lines 242-257):
```rust
UnifiedRequest::Image(image_req) => {
    // Handle image inference (currently incomplete)
    log_tx_clone.send("Processing image inference...".to_string()).ok();
    // TODO: Implement image inference
    // let result = instance.infer(image_req.tensor_bytes)?;
    let response_data = ApiResponseData::Image { 
        label: 0, // TODO: Actual inference
        probability: 0.0 
    };
    let _ = image_req.responder.send(response_data);
    Ok(())
}
```

**New Working Code** (using cascadia-demo runtime):
```rust
// Initialize cascadia-demo runtime at server startup
static WASM_INSTANCE: std::sync::OnceLock<std::sync::Arc<std::sync::Mutex<runtime::WasmInstance>>> = std::sync::OnceLock::new();

// In server initialization:
let engine = Arc::new(Engine::new(&Config::new()).unwrap());
let module = Arc::new(Module::from_file(&engine, Path::new("../target/wasm32-wasip1/debug/inferencer.wasm")).unwrap());
let instance = Arc::new(Mutex::new(runtime::WasmInstance::new(engine, module)?));
WASM_INSTANCE.set(instance).unwrap();

// Request handler with JSON API adapter:
UnifiedRequest::Image(image_req) => {
    log_tx_clone.send("Processing image inference...".to_string()).ok();
    
    // JSON API Adapter: base64 → JPEG bytes
    let jpeg_bytes = base64::decode(&image_req.image_data)
        .map_err(|e| anyhow!("Invalid base64: {}", e))?;
    
    // Use cascadia-demo's proven preprocessing  
    let tensor_bytes = tensor::jpeg_to_raw_bgr(jpeg_bytes, &log_tx_clone)?;
    
    // Use cascadia-demo's proven inference
    let instance = WASM_INSTANCE.get().unwrap();
    let mut wasm_instance = instance.lock().unwrap();
    let result = wasm_instance.infer(tensor_bytes)?;
    
    let response_data = ApiResponseData::Image { 
        label: result.0, 
        probability: result.1 
    };
    let _ = image_req.responder.send(response_data);
    Ok(())
}
```

**Key Changes**:
- Add base64 decoding adapter for JSON API compatibility
- Integrate cascadia-demo's `WasmInstance` and `tensor::jpeg_to_raw_bgr`
- Preserve current module's request/response structures
- Use cascadia-demo's exact inference flow

### Step 1.3: Update Build Configuration

**File**: `module/Cargo.toml`

Replace current workspace dependencies with cascadia-demo's proven ones:
```toml
# Copy from cascadia-demo/Cargo.toml exactly
[workspace]
members = ["server", "inferencer"]
resolver = "2"
```

**File**: `module/inferencer/Cargo.toml`
```toml
# Copy from cascadia-demo/inferencer/Cargo.toml exactly
[package]
name = "inferencer"
version = "0.0.0"
authors = ["Saam Tehrani"]
edition = "2021"

[dependencies]
wasi-nn = "0.1.0"
log = "0.4"
wasm-logger = "0.2"
env_logger = "0.10"
```

**File**: `module/server/Cargo.toml`
```toml
# Copy from cascadia-demo/server/Cargo.toml, add base64 for JSON adapter
[dependencies]
# ... all cascadia-demo dependencies exactly ...
base64 = "0.21"  # Add for JSON API adapter
```

**File**: `module/Makefile`
```makefile
# Copy cascadia-demo build process exactly
inferencer-build: 
	cargo build --package inferencer --target wasm32-wasip1

server-build:
	cargo build --package server

dev: inferencer-build server-build
	cargo run --package server

clean:
	cargo clean
```

### Step 1.4: Frontend Compatibility

**Current frontend already compatible**: The existing `module/frontend/app.js` sends:
```javascript
{
  "model": "squeezenet1.1-7", 
  "image": "base64_jpeg_data"
}
```

**Adapter needed in server**: Change model name handling:
```rust
// In request parsing - adapt model names
let model_name = match api_request.model.as_str() {
    "squeezenet1.1-7" | "mobilenet-v3-large" => "mobilenet_v2_1.0_224", // cascadia-demo's model
    _ => return Err(anyhow!("Unsupported model: {}", api_request.model))
};
```

### Step 1.5: Validation Checklist

**Phase 1 Success Criteria**:
- [ ] `make dev` builds without errors
- [ ] Server starts and loads WASM module successfully  
- [ ] Frontend image clicks return classification results
- [ ] Results are reasonable (not 0/0.0 placeholders)
- [ ] Performance similar to cascadia-demo baseline

**If Phase 1 fails**: Debug cascadia-demo integration, not Component Model issues.

## Phase 2: Refactor to Component Model

### Goal: Replace FFI with Component Model While Preserving Functionality

This phase modernizes the working FFI integration to Component Model, with the confidence of a working baseline to compare against.

### Code to be Rewritten in Phase 2

#### 2.1: WASM Interface (Complete Rewrite)

**File**: `module/inferencer/src/lib.rs`

**Current (Phase 1 - FFI exports)**:
```rust
#[no_mangle]
pub extern "C" fn infer(tensor_ptr: i32, tensor_len: i32, result_ptr: i32) -> i32 {
    // Manual memory management, unsafe pointers
}

static MODEL: OnceLock<MobilnetModel> = OnceLock::new();
```

**Target (Phase 2 - Component Model)**:
```rust
use wit_bindgen::generate;

generate!({
    world: "ncl-ml-component",
    path: "../wit",
});

struct Component;

impl exports::ncl::ml::image_classifier::Guest for Component {
    fn classify(tensor_data: Vec<u8>) -> Result<ClassificationResult, ClassificationError> {
        // Same inference logic, but type-safe interface
        let model = MODEL.get().ok_or("Model not loaded")?;
        let tensor = model.tensor_from_raw_data(&tensor_data)?;
        let result = model.run_inference(tensor)?;
        
        Ok(ClassificationResult {
            label: result.0,
            probability: result.1,
        })
    }
}

// Same MODEL static, same inference logic - just wrapped in Component interface
static MODEL: OnceLock<MobilnetModel> = OnceLock::new();
```

**Rewrite Strategy**: Keep all inference logic identical, replace only the interface layer.

#### 2.2: WIT Interface Definition (New File)

**File**: `module/inferencer/wit/ncl-ml.wit` (New)
```wit
package ncl:ml@0.1.0;

interface image-classifier {
    record classification-result {
        label: u32,
        probability: float32,
    }
    
    variant classification-error {
        model-not-loaded(string),
        invalid-tensor(string),
        inference-failed(string),
    }
    
    classify: func(tensor-data: list<u8>) -> result<classification-result, classification-error>;
    load-model: func() -> result<unit, string>;
    is-ready: func() -> bool;
}

world ncl-ml-component {
    export image-classifier;
    import wasi:filesystem/types@0.2.0;
    import wasi:filesystem/preopens@0.2.0;
}
```

#### 2.3: Server Runtime (Significant Rewrite)

**File**: `module/server/src/runtime.rs`

**Current (Phase 1 - Traditional WASM)**:
```rust
pub struct WasmInstance {
    instance: Instance,
    store: Store<Context>,
    memory: Memory,
}

impl WasmInstance {
    pub fn infer(&mut self, tensor_bytes: Vec<u8>) -> anyhow::Result<InferenceResult> {
        // Manual memory management, FFI calls
        let ptr = 1008;
        self.memory.write(&mut self.store, ptr, &tensor_bytes)?;
        let infer = self.instance.get_typed_func::<(i32, i32, i32), ()>(&mut self.store, "infer")?;
        infer.call(&mut self.store, (ptr as i32, tensor_bytes.len() as i32, result_ptr))?;
        // Manual result extraction...
    }
}
```

**Target (Phase 2 - Component Model)**:
```rust
use wasmtime::component::{Component, Linker, bindgen};

bindgen!({
    world: "ncl-ml-component",
    path: "../inferencer/wit",
});

pub struct ComponentInstance {
    store: Store<Context>,
    interface: NclMlComponent,
}

impl ComponentInstance {
    pub fn classify(&mut self, tensor_data: Vec<u8>) -> anyhow::Result<(u32, f32)> {
        // Type-safe calls, no manual memory management
        let result = self.interface
            .ncl_ml_image_classifier()
            .call_classify(&mut self.store, &tensor_data)?
            .map_err(|e| anyhow::anyhow!(e))?;
            
        Ok((result.label, result.probability))
    }
}

// Same Context setup, same OpenVINO backend - just Component instantiation
```

**Rewrite Strategy**: Replace manual memory management with Component Model's type-safe interface, but preserve all OpenVINO setup and context management.

#### 2.4: Build Configuration (Moderate Changes)

**File**: `module/inferencer/Cargo.toml`

**Current (Phase 1)**:
```toml
[dependencies]
wasi-nn = "0.1.0"
log = "0.4"
wasm-logger = "0.2"
```

**Target (Phase 2)**:
```toml
[dependencies]
wit-bindgen = "0.19"
wasi-nn = "0.1.0"  # Same
log = "0.4"        # Same
wasm-logger = "0.2" # Same

[lib]
crate-type = ["cdylib"]

[package.metadata.component]
package = "ncl:ml"
```

**File**: `module/Makefile`

**Current (Phase 1)**:
```makefile
inferencer-build: 
	cargo build --package inferencer --target wasm32-wasip1
```

**Target (Phase 2)**:
```makefile
inferencer-build: 
	cargo component build --package inferencer --release
```

### Code to Remain Unchanged in Phase 2

#### 2.1: Image Processing (No Changes)
**File**: `module/server/src/tensor.rs`
- Keep cascadia-demo's `jpeg_to_raw_bgr()` exactly
- Same OpenCV processing pipeline
- Same tensor format and validation

#### 2.2: Server Infrastructure (Minimal Changes)
**File**: `module/server/src/lib.rs`
- Keep JSON API adapter logic
- Keep request/response structures  
- Change only: `instance.infer()` → `component.classify()`

#### 2.3: Frontend (No Changes)
- Same JSON API
- Same image gallery
- Same result display

#### 2.4: Model Files (No Changes)
- Same `model.xml` and `model.bin`
- Same fixture directory structure

### Phase 2 Implementation Steps

#### 2.1: Create WIT Interface
1. Create `module/inferencer/wit/ncl-ml.wit`
2. Test compilation with `wit-bindgen`

#### 2.2: Rewrite Component Interface
1. Replace FFI exports with Component Model in `inferencer/src/lib.rs`
2. Preserve all inference logic, change only interface layer
3. Test component builds with `cargo component build`

#### 2.3: Update Server Runtime
1. Replace `WasmInstance` with `ComponentInstance`
2. Keep same OpenVINO setup and context management
3. Replace FFI calls with Component interface calls

#### 2.4: Update Build Process
1. Add Component Model dependencies
2. Update Makefile to use `cargo component build`
3. Verify end-to-end build works

#### 2.5: Integration Testing
1. Compare Phase 2 results against Phase 1 baseline
2. Verify same accuracy and performance
3. Test error handling improvements

### Phase 2 Validation Checklist

**Success Criteria**:
- [ ] Component builds successfully with `cargo component build`
- [ ] Classification results identical to Phase 1 baseline
- [ ] Performance comparable to Phase 1 baseline  
- [ ] Error handling improved (structured error types)
- [ ] Type safety verified (no `unsafe` in interface layer)

**Failure Plan**: If Component Model integration fails, revert to Phase 1 FFI approach (working fallback).

## Risk Mitigation Strategy

### Phase 1 Risks (Low)
- **Integration complexity**: Mitigated by copying proven code exactly
- **API incompatibility**: Mitigated by simple base64 adapter pattern
- **Build issues**: Mitigated by copying cascadia-demo's exact dependencies

### Phase 2 Risks (Medium)
- **Component Model compatibility**: Mitigated by having working Phase 1 baseline
- **Performance regression**: Mitigated by direct comparison against Phase 1
- **Interface complexity**: Mitigated by preserving all core inference logic

### Fallback Strategy
If Phase 2 fails, Phase 1 provides a fully functional image classification system using proven cascadia-demo technology. This is an acceptable fallback that meets all functional requirements.

## Timeline and Effort Estimates

### Phase 1: Integration (2-4 hours)
- File copying and basic adaptation: 1 hour
- JSON API adapter implementation: 1 hour  
- Build configuration and testing: 1-2 hours
- **Total**: 3-4 hours for working system

### Phase 2: Component Model Refactor (1-2 days)
- WIT interface design and testing: 2-4 hours
- Component implementation: 4-6 hours
- Server integration: 2-4 hours
- Testing and validation: 2-4 hours
- **Total**: 10-18 hours for modernized architecture

### Total Project: 2-3 days
- Day 1: Phase 1 completion (working state)
- Days 2-3: Phase 2 implementation (modern architecture)

## Success Criteria

### Phase 1 Success (Required)
- [ ] Image classification working end-to-end
- [ ] Results comparable to cascadia-demo accuracy
- [ ] JSON API compatible with existing frontend
- [ ] Performance within 50% of cascadia-demo baseline

### Phase 2 Success (Desired)  
- [ ] Component Model interface working
- [ ] Type safety improvements verified
- [ ] Performance maintained from Phase 1
- [ ] Error handling enhanced
- [ ] Architecture modernized and maintainable

This two-phase approach provides confidence through incremental progress while minimizing the risk of Component Model compatibility issues derailing the entire project.