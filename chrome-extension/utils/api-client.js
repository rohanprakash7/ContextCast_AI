/**
 * API Client for Backend Communication
 * Handles all HTTP requests to FastAPI backend
 */

const API_BASE_URL = 'http://localhost:8000';

/**
 * Check API health
 * @returns {Promise<Object>} Health status
 */
async function checkHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return await response.json();
}

/**
 * Check if video is already indexed
 * @param {string} videoId - YouTube video ID
 * @returns {Promise<Object>} Status information
 */
async function checkVideoStatus(videoId) {
    const response = await fetch(`${API_BASE_URL}/index/status/${videoId}`);
    return await response.json();
}

/**
 * Index a YouTube video
 * @param {string} videoId - YouTube video ID
 * @returns {Promise<Object>} Indexing results
 */
async function indexVideo(videoId) {
    const response = await fetch(`${API_BASE_URL}/index`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            video_id: videoId
        })
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Indexing failed');
    }
    
    return await response.json();
}

/**
 * Ask a question about video content
 * @param {string} question - User's question
 * @param {string} videoId - YouTube video ID
 * @param {Object} options - Query options
 * @returns {Promise<Object>} Answer with sources
 */
async function askQuestion(question, videoId, options = {}) {
    const requestBody = {
        question: question,
        video_id: videoId,
        retriever_type: options.retrieverType || 'simple',
        top_k: options.topK || 4,
        include_citations: options.includeCitations !== false
    };
    
    const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Query failed');
    }
    
    return await response.json();
}