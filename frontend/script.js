const API_BASE = "http://localhost:8000";

const youtubeUrlInput = document.getElementById("youtubeUrl");
const questionInput   = document.getElementById("questionInput");
const indexBtn        = document.getElementById("indexBtn");
const askBtn          = document.getElementById("askBtn");
const clearBtn        = document.getElementById("clearBtn");
const indexStatus     = document.getElementById("indexStatus");
const chatHistory     = document.getElementById("chatHistory");
const retrieverSelect = document.getElementById("retrieverType");

// In-memory conversation history sent to backend
let conversationHistory = [];

// ── Helpers ──────────────────────────────────────────────

function extractVideoId(url) {
  try {
    const parsed = new URL(url);
    if (parsed.hostname.includes("youtube.com")) return parsed.searchParams.get("v");
    if (parsed.hostname.includes("youtu.be"))   return parsed.pathname.replace("/", "");
    return null;
  } catch {
    return null;
  }
}

function setStatus(el, text, type = "") {
  el.textContent = text;
  el.className = "status " + type;
}

function scrollToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function removeWelcome() {
  const welcome = chatHistory.querySelector(".welcome-message");
  if (welcome) welcome.remove();
}

// ── Render helpers ───────────────────────────────────────

function appendUserMessage(text) {
  removeWelcome();
  const div = document.createElement("div");
  div.className = "message user";
  div.innerHTML = `
    <div class="avatar">You</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  chatHistory.appendChild(div);
  scrollToBottom();
  return div;
}

function appendAssistantBubble() {
  const groupEl = document.createElement("div");
  groupEl.className = "message-group";

  const msgDiv = document.createElement("div");
  msgDiv.className = "message assistant";

  const bubbleEl = document.createElement("div");
  bubbleEl.className = "bubble streaming";
  bubbleEl.innerHTML = '<span class="cursor">▍</span>';

  msgDiv.innerHTML = '<div class="avatar">AI</div>';
  msgDiv.appendChild(bubbleEl);
  groupEl.appendChild(msgDiv);
  chatHistory.appendChild(groupEl);
  scrollToBottom();
  return { groupEl, bubbleEl };
}

function appendSources(groupEl, sources) {
  if (!sources || sources.length === 0) return;

  const details = document.createElement("details");
  details.className = "sources-toggle";
  details.innerHTML = `
    <summary>${sources.length} source chunk${sources.length > 1 ? "s" : ""}</summary>
    <div class="source-list">
      ${sources.map(s => `
        <div class="source-item">
          <strong>Chunk ${s.chunk_id}</strong>
          ${escapeHtml(s.text)}
        </div>
      `).join("")}
    </div>
  `;
  groupEl.appendChild(details);
  scrollToBottom();
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatAnswer(text) {
  // Bold **text**, preserve newlines
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

// ── Index video ──────────────────────────────────────────

async function indexVideo() {
  const url     = youtubeUrlInput.value.trim();
  const videoId = extractVideoId(url);

  if (!videoId) {
    setStatus(indexStatus, "Invalid YouTube URL.", "error");
    return;
  }

  indexBtn.disabled = true;
  setStatus(indexStatus, "Indexing video...");

  try {
    const response = await fetch(`${API_BASE}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId, namespace: "", chunk_size: 1000, chunk_overlap: 200 })
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Indexing failed");

    setStatus(indexStatus, `Indexed ${data.num_chunks} chunks`, "success");
    // Reset conversation when a new video is indexed
    conversationHistory = [];
    chatHistory.innerHTML = "";
    const welcome = document.createElement("div");
    welcome.className = "welcome-message";
    welcome.innerHTML = `<div class="welcome-icon">✅</div><h2>Video indexed!</h2><p>Now ask any question about the video.</p>`;
    chatHistory.appendChild(welcome);
  } catch (err) {
    setStatus(indexStatus, `Error: ${err.message}`, "error");
  } finally {
    indexBtn.disabled = false;
  }
}

// ── Ask question (streaming) ──────────────────────────────

async function askQuestion() {
  const url      = youtubeUrlInput.value.trim();
  const question = questionInput.value.trim();
  const videoId  = extractVideoId(url);

  if (!videoId) { alert("Please enter a valid YouTube URL first."); return; }
  if (!question) return;

  questionInput.value = "";
  questionInput.style.height = "auto";
  askBtn.disabled = true;

  removeWelcome();
  appendUserMessage(question);

  // Create assistant bubble immediately (empty, no typing indicator needed)
  const { groupEl, bubbleEl } = appendAssistantBubble();

  let fullAnswer = "";

  try {
    const response = await fetch(`${API_BASE}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        video_id: videoId,
        retriever_type: retrieverSelect.value,
        top_k: 6,
        include_citations: true,
        chat_history: conversationHistory
      })
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Stream failed");
    }

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by double newlines
      const events = buffer.split("\n\n");
      buffer = events.pop(); // keep incomplete tail

      for (const event of events) {
        if (!event.startsWith("data: ")) continue;
        const raw = event.slice(6).trim();

        if (raw === "[DONE]") break;

        if (raw.startsWith("[SOURCES]")) {
          const payload = JSON.parse(raw.slice(9));
          appendSources(groupEl, payload.sources);
          continue;
        }

        // Regular token
        const { token } = JSON.parse(raw);
        fullAnswer += token;
        bubbleEl.innerHTML = formatAnswer(fullAnswer);
        scrollToBottom();
      }
    }

    // Remove streaming cursor
    bubbleEl.classList.remove("streaming");

    // Save to conversation history for next turn
    conversationHistory.push({ question, answer: fullAnswer });

  } catch (err) {
    bubbleEl.textContent = `Error: ${err.message}`;
  } finally {
    askBtn.disabled = false;
  }
}

// ── Clear conversation ────────────────────────────────────

function clearConversation() {
  conversationHistory = [];
  chatHistory.innerHTML = `
    <div class="welcome-message">
      <div class="welcome-icon">🎬</div>
      <h2>Ask about any YouTube video</h2>
      <p>Index a video on the left, then start chatting below.</p>
    </div>
  `;
}

// ── Auto-resize textarea ──────────────────────────────────

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = questionInput.scrollHeight + "px";
});

// Submit on Enter (Shift+Enter for newline)
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
});

// ── Event listeners ───────────────────────────────────────

indexBtn.addEventListener("click", indexVideo);
askBtn.addEventListener("click", askQuestion);
clearBtn.addEventListener("click", clearConversation);
