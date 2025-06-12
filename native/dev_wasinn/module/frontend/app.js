const titleElement = document.querySelector("title");
const titleText = "Intel + Community Labs = ML + AO = ";
const displayLength = 23;
let currentIndex = 0;

function rotateTitle() {
    let rotatedText = titleText + titleText;
    let displayText = rotatedText.substring(currentIndex, currentIndex + displayLength);
    titleElement.textContent = displayText;
    currentIndex = (currentIndex + 1) % titleText.length;
}

setInterval(rotateTitle, 500);

const serverURL = "http://127.0.0.1:3000";

const textModeBtn = document.getElementById('text-mode-btn');
const imageModeBtn = document.getElementById('image-mode-btn');
const textModeContent = document.getElementById('text-mode-content');
const imageModeContent = document.getElementById('image-mode-content');

textModeBtn.addEventListener('click', () => {
    textModeBtn.classList.add('active');
    imageModeBtn.classList.remove('active');
    textModeContent.style.display = 'block';
    imageModeContent.style.display = 'none';
});

imageModeBtn.addEventListener('click', () => {
    imageModeBtn.classList.add('active');
    textModeBtn.classList.remove('active');
    imageModeContent.style.display = 'block';
    textModeContent.style.display = 'none';
});

const chatForm = document.getElementById('chat-form');
const chatHistory = document.getElementById('chat-history');
const userPromptInput = document.getElementById('user-prompt');

function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${isUser ? 'user-message' : 'assistant-message'}`;
    messageDiv.textContent = content;
    
    const welcomeMessage = chatHistory.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    return messageDiv;
}

chatForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const userMessage = userPromptInput.value.trim();
    if (!userMessage) return;
    
    const maxTokensInput = document.getElementById('max-tokens');
    const maxTokens = parseInt(maxTokensInput.value) || 50;
    
    logElement.innerHTML = '';
    
    addMessage(userMessage, true);
    userPromptInput.value = '';
    
    // Create response message with animated loading dots
    const responseMessage = addMessage('', false);
    let responseText = '';
    let isStreaming = true;
    
    // Get model name from the request
    const modelName = 'llama3.1-8b-instruct';
    
    // Create typing indicator
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.textContent = `${modelName} is typing...`;
    typingIndicator.style.cssText = 'font-size: 12px; color: #666; margin-top: 2px; opacity: 0.7;';
    responseMessage.parentNode.appendChild(typingIndicator);
    
    const streamEventSource = new EventSource(serverURL + "/logs");
    
    streamEventSource.onmessage = (event) => {
        if (event.data.startsWith('[TEXT_TOKEN]')) {
            const tokenText = event.data.substring('[TEXT_TOKEN]'.length);
            responseText += tokenText;
            responseMessage.textContent = responseText;
        }
        else if (event.data === '[TEXT_DONE]') {
            isStreaming = false;
            streamEventSource.close();
            typingIndicator.remove();
            if (!responseText) {
                responseMessage.textContent = 'No response generated';
            }
        }
    };
    
    streamEventSource.onerror = (error) => {
        console.error('Streaming error:', error);
        isStreaming = false;
        streamEventSource.close();
        typingIndicator.remove();
        if (!responseText) {
            responseMessage.textContent = 'Error: Could not stream response';
        }
    };
    
    try {
        const response = await fetch(`${serverURL}/infer`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: modelName,
                prompt: userMessage,
                max_tokens: maxTokens
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        await response.json();
        
    } catch (error) {
        console.error('Error:', error);
        isStreaming = false;
        streamEventSource.close();
        typingIndicator.remove();
        responseMessage.textContent = 'Error: Could not connect to server';
    }
});

const outputElement = document.querySelector('.model-output');
const logElement = document.querySelector('.under-the-hood-logs');

const eventSource = new EventSource(serverURL + "/logs");

eventSource.onopen = (event) => {
    logElement.innerHTML += `${new Date().toLocaleTimeString('en-US', {hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'})}.${new Date().getMilliseconds().toString().padStart(3, '0').slice(0, 2)}: SSE connection opened.<br>`;
    console.log("SSE connection opened:", event);
};

eventSource.onmessage = (event) => {
    logElement.innerHTML = `${new Date().toLocaleTimeString('en-US', {hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'})}.${new Date().getMilliseconds().toString().padStart(3, '0').slice(0, 2)}: ${event.data}<br>` + logElement.innerHTML;
    logElement.scrollTop = logElement.scrollHeight;
};

eventSource.onerror = (err) => {
    console.error("SSE Error:", err);
};

document.querySelectorAll('.gallery img').forEach(img => {
    img.addEventListener('click', async function() {
        logElement.innerHTML = '';
        outputElement.innerHTML = `Processing...`;

        logElement.innerHTML += `${new Date().toLocaleTimeString('en-US', {hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'})}.${new Date().getMilliseconds().toString().padStart(3, '0').slice(0, 2)}: [frontend/app.js] Sending inference request for ${this.alt || 'image'}.<br>`;

        try {
            const response = await fetch(this.src);
            const blob = await response.blob();
            
            const base64 = await new Promise((resolve) => {
                const reader = new FileReader();
                reader.onload = () => {
                    // Remove the "data:image/jpeg;base64," prefix
                    const base64 = reader.result.split(',')[1];
                    resolve(base64);
                };
                reader.readAsDataURL(blob);
            });
            
            const modelName = 'squeezenet1.1-7';
            const serverResponse = await fetch(serverURL + "/infer", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: modelName,
                    image: base64
                })
            });
            
            if (!serverResponse.ok) {
                const errorText = await serverResponse.text();
                throw new Error(`Server responded with status ${serverResponse.status}: ${errorText}`);
            }

            const result = await serverResponse.json();
            const label = result?.label ?? 'unknown';
            const confidence = result?.probability ?? 0;

            outputElement.innerHTML = `${modelName} identified a ${label} with ${(confidence*100).toFixed(2)}% confidence.`;
            
        } catch (error) {
            console.error(error);
            outputElement.innerHTML = `Error processing image: ${error.message || error}. Please try again.`;
        }
    });
});

const lastImage = document.querySelector('.gallery img:last-child');
lastImage.addEventListener('dragover', (e) => {
    e.preventDefault();
    lastImage.style.border = '2px solid green';
});

lastImage.addEventListener('dragleave', () => {
    lastImage.style.border = '2px solid red';
});

lastImage.addEventListener('drop', async (e) => {
    e.preventDefault();
    lastImage.style.border = '2px solid transparent';
    
    const file = e.dataTransfer.files[0];
    if (file.type !== 'image/jpeg' && file.type !== 'image/jpg') {
        alert('Please drop a JPEG/JPG image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        lastImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
});