const API_BASE = "http://localhost:8000";

function getVideoId() {
  const url = new URL(window.location.href);
  return url.searchParams.get("v");
}

function getVideoTitle() {
  const titleEl =
    document.querySelector("ytd-watch-metadata h1 yt-formatted-string") ||
    document.querySelector("h1.title yt-formatted-string") ||
    document.querySelector("h1");
  return titleEl ? titleEl.textContent.trim() : "Current YouTube Video";
}

function createContextCastUI() {
  if (document.getElementById("contextcast-root")) return;

  const root = document.createElement("div");
  root.id = "contextcast-root";

  root.innerHTML = `
    <div class="cc-panel">
      <div class="cc-header">
        <div class="cc-brand">ContextCast AI</div>
        <div class="cc-status-line">
          <span class="cc-dot"></span>
          <span>RAG Ready</span>
        </div>
      </div>

      <div class="cc-body">
        <div class="cc-video-title" id="cc-video-title"></div>
        <div class="cc-video-id" id="cc-video-id"></div>

        <div class="cc-index-banner" id="cc-index-banner">
          Video indexed! Ask me anything about this video.
        </div>

        <div class="cc-chat" id="cc-chat"></div>
      </div>

      <div class="cc-input-wrap">
        <textarea
          id="cc-question"
          class="cc-question"
          placeholder="what is this video about?"
          rows="3"
        ></textarea>
        <button id="cc-ask-btn" class="cc-ask-btn">Ask</button>
      </div>

      <div class="cc-footer">Powered by LangChain</div>
    </div>
  `;

  document.body.appendChild(root);

  const titleEl = document.getElementById("cc-video-title");
  const idEl = document.getElementById("cc-video-id");
  const askBtn = document.getElementById("cc-ask-btn");
  const questionEl = document.getElementById("cc-question");

  if (titleEl) titleEl.textContent = getVideoTitle();
  if (idEl) idEl.textContent = `VID: ${getVideoId() || "N/A"}`;

  if (askBtn) {
    askBtn.addEventListener("click", askQuestion);
  }

  if (questionEl) {
    questionEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
      }
    });
  }

  autoIndexVideo();
}

function appendMessage(role, text) {
  const chat = document.getElementById("cc-chat");
  if (!chat) return;

  const msg = document.createElement("div");
  msg.className = `cc-msg ${role}`;
  msg.textContent = text;

  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

async function autoIndexVideo() {
  const videoId = getVideoId();
  const banner = document.getElementById("cc-index-banner");

  if (!videoId) {
    if (banner) banner.textContent = "Could not detect YouTube video.";
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/index`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        video_id: videoId,
        namespace: "",
        chunk_size: 1000,
        chunk_overlap: 200
      })
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Indexing failed");
    }

    if (banner) {
      banner.textContent =
        data.status === "already_indexed"
          ? "Video indexed! Ask me anything about this video."
          : `Video indexed! ${data.num_chunks || 0} chunks ready.`;
    }
  } catch (err) {
    if (banner) banner.textContent = `Indexing error: ${err.message}`;
  }
}

async function askQuestion() {
  const videoId = getVideoId();
  const questionEl = document.getElementById("cc-question");
  const question = questionEl ? questionEl.value.trim() : "";

  if (!videoId || !question) return;

  appendMessage("user", question);
  if (questionEl) questionEl.value = "";

  const loadingMsg = document.createElement("div");
  loadingMsg.className = "cc-msg assistant";
  loadingMsg.id = "cc-loading";
  loadingMsg.textContent = "Thinking...";

  const chat = document.getElementById("cc-chat");
  if (chat) chat.appendChild(loadingMsg);

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question,
        video_id: videoId,
        retriever_type: "simple",
        top_k: 6,
        include_citations: true
      })
    });

    const data = await res.json();

    const loading = document.getElementById("cc-loading");
    if (loading) loading.remove();

    if (!res.ok) {
      throw new Error(data.detail || "Query failed");
    }

    appendMessage("assistant", data.answer || "No answer returned.");
  } catch (err) {
    const loading = document.getElementById("cc-loading");
    if (loading) loading.remove();
    appendMessage("assistant", `Error: ${err.message}`);
  }
}

createContextCastUI();