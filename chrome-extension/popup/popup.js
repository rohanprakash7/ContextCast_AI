/**
 * Popup Script - Main Extension Logic
 * Handles user interactions and API communication
 */

let currentVideoId = null;
let isIndexed = false;

// DOM Elements
const videoTitleDiv = document.getElementById('video-title');
const videoIdDiv = document.getElementById('video-id');
const notYoutubeDiv = document.getElementById('not-youtube');
const indexSection = document.getElementById('index-section');
const indexStatusDiv = document.getElementById('index-status');
const indexBtn = document.getElementById('index-btn');
const indexProgress = document.getElementById('index-progress');
const chatSection = document.getElementById('chat-section');
const chatMessages = document.getElementById('chat-messages');
const questionInput = document.getElementById('question-input');
const askBtn = document.getElementById('ask-btn');
const loadingDiv = document.getElementById('loading');
const apiStatusDot = document.querySelector('.status-dot');
const apiStatusText = document.querySelector('.status-text');

/**
 * Initialize extension on popup open
 */
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Popup loaded');
    
    // Check API health
    await checkAPIHealth();
    
    // Get current video info
    await loadVideoInfo();
    
    // Setup event listeners
    setupEventListeners();
});

/**
 * Check if backend API is healthy
 */
async function checkAPIHealth() {
    try {
        const health = await checkHealth();
        
        if (health.status === 'healthy') {
            apiStatusDot.classList.remove('error');
            apiStatusText.textContent = 'Connected';
        }
    } catch (error) {
        apiStatusDot.classList.add('error');
        apiStatusText.textContent = 'API Offline';
        console.error('API health check failed:', error);
    }
}

/**
 * Load current video information
 */
async function loadVideoInfo() {
    try {
        // Get current tab
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        // Send message to content script
        const response = await chrome.tabs.sendMessage(tab.id, { action: 'getVideoInfo' });
        
        if (response && response.isYouTubePage && response.videoId) {
            // On YouTube video page
            currentVideoId = response.videoId;
            
            videoTitleDiv.textContent = response.videoTitle;
            videoIdDiv.textContent = `ID: ${response.videoId}`;
            
            notYoutubeDiv.style.display = 'none';
            
            // Check if already indexed
            await checkIndexStatus();
        } else {
            // Not on YouTube
            notYoutubeDiv.style.display = 'block';
            indexSection.style.display = 'none';
            chatSection.style.display = 'none';
        }
    } catch (error) {
        console.error('Failed to load video info:', error);
        notYoutubeDiv.style.display = 'block';
    }
}

/**
 * Check if current video is indexed
 */
async function checkIndexStatus() {
    try {
        indexStatusDiv.textContent = 'Checking index status...';
        
        const status = await checkVideoStatus(currentVideoId);
        
        isIndexed = status.is_indexed;
        
        if (isIndexed) {
            // Video already indexed - show chat
            indexSection.style.display = 'none';
            chatSection.style.display = 'flex';
        } else {
            // Video not indexed - show index button
            indexSection.style.display = 'block';
            chatSection.style.display = 'none';
            indexStatusDiv.textContent = 'Video not indexed yet';
        }
    } catch (error) {
        console.error('Failed to check index status:', error);
        indexStatusDiv.textContent = 'Could not check status';
        indexSection.style.display = 'block';
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Index button
    indexBtn.addEventListener('click', handleIndexVideo);
    
    // Ask button
    askBtn.addEventListener('click', handleAskQuestion);
    
    // Enter key in input
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleAskQuestion();
        }
    });
}

/**
 * Handle index video button click
 */
async function handleIndexVideo() {
    try {
        indexBtn.disabled = true;
        indexProgress.style.display = 'block';
        indexStatusDiv.textContent = 'Indexing...';
        
        const result = await indexVideo(currentVideoId);
        
        indexProgress.style.display = 'none';
        indexStatusDiv.textContent = `Indexed! ${result.num_chunks} chunks created`;
        
        // Switch to chat interface
        setTimeout(() => {
            indexSection.style.display = 'none';
            chatSection.style.display = 'flex';
            isIndexed = true;
        }, 1500);
        
    } catch (error) {
        console.error('Indexing failed:', error);
        indexBtn.disabled = false;
        indexProgress.style.display = 'none';
        indexStatusDiv.textContent = 'Indexing failed. Try again.';
        alert(`Failed to index video: ${error.message}`);
    }
}

/**
 * Handle ask question button click
 */
async function handleAskQuestion() {
    const question = questionInput.value.trim();
    
    if (!question || question.length < 3) {
        return;
    }
    
    try {
        // Add user message
        addMessage('user', question);
        
        // Clear input
        questionInput.value = '';
        
        // Disable input while processing
        askBtn.disabled = true;
        questionInput.disabled = true;
        loadingDiv.style.display = 'flex';
        
        // Call API
        const result = await askQuestion(question, currentVideoId, {
            retrieverType: 'simple',
            topK: 3,
            includeCitations: false
        });
        
        // Add assistant message
        addMessage('assistant', result.answer);
        
    } catch (error) {
        console.error('Query failed:', error);
        addMessage('error', `Error: ${error.message}`);
    } finally {
        // Re-enable input
        askBtn.disabled = false;
        questionInput.disabled = false;
        loadingDiv.style.display = 'none';
        questionInput.focus();
    }
}

/**
 * Add message to chat
 */
function addMessage(type, content, metadata = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = `${type}-message`;
    bubbleDiv.textContent = content;
    
    // Add metadata for assistant messages
    if (type === 'assistant' && (metadata.chunks || metadata.duration)) {
        const metaDiv = document.createElement('div');
        metaDiv.style.fontSize = '11px';
        metaDiv.style.color = '#6b7280';
        metaDiv.style.marginTop = '6px';
        metaDiv.textContent = `${metadata.chunks || 0} chunks • ${(metadata.duration || 0).toFixed(1)}s`;
        bubbleDiv.appendChild(metaDiv);
    }
    
    messageDiv.appendChild(bubbleDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom smoothly
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 100);
}