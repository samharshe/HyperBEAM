# dev_wasinn HTTP API Refactor Plan

## Executive Summary

This document outlines a precise, step-by-step plan to refactor the `dev_wasinn` module to expose ML inference capabilities via a clean HTTP API. The refactor will enable both text (LLM) and image (classification) inference through a unified interface while making minimal changes to the existing fragile components.

## Current State Analysis

### What's Working
- **Image Inference**: HTTP POST to `/infer` with raw JPEG data works correctly
- **Frontend**: Has UI for both text and image modes
- **SSE Infrastructure**: `/logs` endpoint provides Server-Sent Events capability
- **WASM Components**: Image classification models load and execute properly

### What's Broken
- **Text Inference**: Server currently runs a hard-coded text inference on startup but doesn't expose HTTP endpoint for text
- **API Consistency**: No unified API format - uses raw data with Content-Type headers
- **Response Format**: Inconsistent between text and image responses

### Critical Constraints
- **DO NOT MODIFY**: WASM initialization, threading model, or model loading mechanisms
- **PRESERVE**: All existing error handling and resource management
- **MINIMAL CHANGES**: Touch only what's necessary to expose the interface

## Architecture Overview

```
Current Flow:
Frontend (app.js) → HTTP Server (lib.rs) → WASM Inferencer → ML Models
                                              ↓
                                         Hard-coded text inference
                                         (needs to be exposed via HTTP)

Target Flow:
Frontend (app.js) → HTTP Server (lib.rs) → WASM Inferencer → ML Models
                          ↓                      ↓
                    /v1/completions         Both text & image
                    (unified JSON API)      inference via HTTP
```

## Implementation Phases

### Phase 1: Fix Text Inference HTTP Handler

**Goal**: Enable text inference via HTTP without breaking existing functionality

**File**: `native/dev_wasinn/module/server/src/lib.rs`

**Steps**:ewhat'

1. **Locate the hard-coded text inference**
   - Find where the server currently runs text inference on startup
   - This will show us the working text inference code

2. **Add text handler to `/infer` endpoint**
   - In the match statement for Content-Type, add:
   ```rust
   "text/plain" => {
       // Extract the text inference code from startup
       // Use the same WASM instance and model
       // Return JSON response like image does
   }
   ```

3. **Remove or comment out the hard-coded startup inference**
   - This frees up the text model for HTTP requests

4. **Test with curl**:
   ```bash
   curl -X POST http://127.0.0.1:3000/infer \
        -H "Content-Type: text/plain" \
        -d "What is the capital of France?"
   ```

**Expected Result**: Server returns JSON with text response (no streaming yet)

### Phase 2: Implement Streaming for Text Responses

**Goal**: Add Server-Sent Events streaming for text generation

**Files**: 
- `native/dev_wasinn/module/server/src/lib.rs`
- `native/dev_wasinn/module/server/src/main.rs`

**Steps**:

1. **Study existing SSE implementation**
   - Look at `/logs` endpoint implementation
   - Note the event stream setup and broadcasting mechanism

2. **Create streaming text endpoint**
   Option A: Modify `/infer` to support streaming
   Option B: Create new `/infer/stream` endpoint
   
   Recommended: Option A with header detection:
   ```rust
   // Check for Accept: text/event-stream header
   if content_type == "text/plain" && accept_header == "text/event-stream" {
       // Return SSE response
   }
   ```

3. **Modify WASM text inference to yield tokens**
   - Look for callback or iterator pattern in text generation
   - Send each token/word as SSE event:
   ```
   data: {"text": "The"}
   data: {"text": " capital"}
   data: {"text": " is"}
   data: {"text": " Paris"}
   data: [DONE]
   ```

4. **Update frontend to handle streaming**
   - Modify `app.js` to use EventSource for text mode
   - Append tokens to chat display as they arrive

**Test**:
```javascript
const eventSource = new EventSource('/infer?stream=true');
eventSource.onmessage = (event) => {
    // Append event.data to chat
};
```

### Phase 3: Create Unified JSON API

**Goal**: Implement OpenAI-style JSON request/response format

**Files**: All server source files in `native/dev_wasinn/module/server/src/`

**Request Format**:
```json
{
  "model": "text-llama3" | "image-squeezenet",
  "data": "text prompt" | "base64_encoded_image_data"
}
```

**Response Format (Image)**:
```json
{
  "result": "cat",
  "confidence": 0.95
}
```

**Response Format (Text - Non-streaming)**:
```json
{
  "result": "The capital of France is Paris."
}
```

**Response Format (Text - Streaming)**:
```
data: {"text": "The"}
data: {"text": " capital"}
data: [DONE]
```

**Implementation Steps**:

1. **Create request/response structs**
   ```rust
   #[derive(Deserialize)]
   struct CompletionRequest {
       model: String,
       data: String,
   }
   ```

2. **Update `/infer` endpoint**
   - Change from Content-Type routing to JSON parsing
   - Route based on `model` field:
     - "text-llama3" → text inference
     - "image-squeezenet" → image inference

3. **Handle base64 image decoding**
   ```rust
   if request.model.starts_with("image") {
       let image_bytes = base64::decode(&request.data)?;
       // Process as before
   }
   ```

4. **Update all responses to JSON format**

### Phase 4: Frontend Updates

**Goal**: Update frontend to use new JSON API

**File**: `native/dev_wasinn/module/frontend/app.js`

**Steps**:

1. **Update image handling**
   - Convert image to base64 before sending
   - Wrap in JSON structure

2. **Update text handling**
   - Wrap text in JSON structure
   - Add streaming support with EventSource

3. **Unify request code**
   ```javascript
   async function callInference(model, data) {
       const response = await fetch('/infer', {
           method: 'POST',
           headers: { 'Content-Type': 'application/json' },
           body: JSON.stringify({ model, data })
       });
       // Handle response
   }
   ```

### Phase 5: Testing & Polish

**Goal**: Ensure everything works correctly

**Test Cases**:

1. **Image Classification**
   - Upload JPEG via UI
   - Drag and drop image
   - Verify correct classification

2. **Text Generation**
   - Type prompt in chat
   - Verify streaming response
   - Check multiple prompts

3. **Error Cases**
   - Invalid model name
   - Malformed base64
   - Empty data

4. **Performance**
   - Monitor for memory leaks
   - Check concurrent requests (though single-user is fine)

## Code Locations & Key Functions

### Server Entry Points
- `native/dev_wasinn/module/server/src/lib.rs:run_server()` - Main server function
- `native/dev_wasinn/module/server/src/lib.rs` - Contains `/infer` endpoint handler
- Look for hard-coded text inference in `main()` or initialization

### Frontend Entry Points
- `native/dev_wasinn/module/frontend/app.js` - All frontend logic
- Image click handler around line 130-145
- Text form submit handler around line 56-91

### WASM Interface
- `native/dev_wasinn/module/inferencer/wit/ncl-ml.wit` - Interface definition
- DO NOT MODIFY - Just understand the available functions

## Common Pitfalls & Solutions

### Pitfall 1: Breaking WASM Instance
**Issue**: Modifying WASM initialization breaks everything
**Solution**: Don't touch any WASM loading code. Only modify HTTP handlers.

### Pitfall 2: Thread Safety
**Issue**: Server runs in separate thread from Erlang NIF
**Solution**: Keep existing threading model. Don't add new threads.

### Pitfall 3: Model Loading
**Issue**: Models are finicky about initialization order
**Solution**: Keep hard-coded model paths and loading sequence.

### Pitfall 4: Memory Management
**Issue**: WASM instances not properly cleaned up
**Solution**: Accept this for demo. Don't try to fix.

## Validation Checklist

After each phase, verify:

- [ ] Existing image inference still works
- [ ] No new compiler warnings
- [ ] Server starts without errors
- [ ] Frontend loads without console errors
- [ ] Manual test passes

## Final Deliverable

A working HTTP API with these capabilities:

1. **Endpoint**: `POST /v1/completions`
2. **Image inference**: Returns classification result
3. **Text inference**: Streams tokens via SSE
4. **Clean JSON API**: Easy to understand and use
5. **Working frontend**: Both modes functional

## Example Usage

### Image Classification
```bash
curl -X POST http://127.0.0.1:3000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "image-squeezenet",
    "data": "base64_encoded_jpeg_data_here"
  }'
```

### Text Generation
```bash
curl -X POST http://127.0.0.1:3000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "text-llama3",
    "data": "What is the meaning of life?"
  }'
```

## Notes for Implementers

1. **Read this entire document before starting**
2. **Test after each phase** - Don't try to do everything at once
3. **When in doubt, make minimal changes**
4. **If something seems fragile, it probably is** - Don't refactor it
5. **Keep the existing structure** - We're adding a facade, not rebuilding

Remember: The goal is a working demo with a clean API, not a production-ready system. Simple and working beats elegant and broken.