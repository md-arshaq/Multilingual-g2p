/**
 * Live TTS Playground Client Logic (tts_app)
 * Manages language presets, live G2P token inspection, and dual GPU audio synthesis.
 */

document.addEventListener("DOMContentLoaded", () => {
  let currentLang = "hi";
  let presetsData = {};

  // DOM Elements
  const langTabs = document.querySelectorAll(".lang-tab-btn");
  const presetContainer = document.getElementById("preset-chips");
  const textInput = document.getElementById("tts-input-text");
  const charCounter = document.getElementById("char-counter");
  const clearBtn = document.getElementById("clear-btn");
  const previewG2pBtn = document.getElementById("preview-g2p-btn");
  const synthesizeBtn = document.getElementById("synthesize-btn");
  
  // G2P Inspector Elements
  const g2pCard = document.getElementById("g2p-inspector-card");
  const reductionBadge = document.getElementById("g2p-reduction-badge");
  const baseCount = document.getElementById("g2p-baseline-count");
  const clustCount = document.getElementById("g2p-clustered-count");
  const baseTokensDiv = document.getElementById("g2p-baseline-tokens");
  const clustTokensDiv = document.getElementById("g2p-clustered-tokens");

  // Synthesis Output Elements
  const resultsArea = document.getElementById("synthesis-results-area");
  const audioBase = document.getElementById("live-audio-baseline");
  const audioClust = document.getElementById("live-audio-clustered");
  const metaBase = document.getElementById("meta-baseline");
  const metaClust = document.getElementById("meta-clustered");
  const downloadBase = document.getElementById("download-baseline-btn");
  const downloadClust = document.getElementById("download-clustered-btn");
  const playSeqBtn = document.getElementById("play-sequence-btn");

  // Load Presets from API
  async function loadPresets() {
    try {
      const res = await fetch("/api/presets");
      const data = await res.json();
      if (data.status === "success") {
        presetsData = data.presets;
        renderPresets(currentLang);
      }
    } catch (err) {
      console.warn("Could not fetch presets", err);
    }
  }

  function renderPresets(lang) {
    const list = presetsData[lang] || [];
    presetContainer.innerHTML = list.map(p => `
      <button type="button" class="chip" data-text="${p.text.replace(/"/g, '&quot;')}">
        ${p.category}: ${p.text.slice(0, 32)}...
      </button>
    `).join("");

    document.querySelectorAll("#preset-chips .chip").forEach(chip => {
      chip.addEventListener("click", () => {
        textInput.value = chip.dataset.text;
        updateCharCount();
        inspectG2P();
      });
    });

    // Auto-fill first preset on initial load if text is empty
    if (!textInput.value.trim() && list.length > 0) {
      textInput.value = list[0].text;
      updateCharCount();
    }
  }

  function updateCharCount() {
    const len = textInput.value.length;
    charCounter.textContent = `${len} char${len !== 1 ? "s" : ""}`;
  }

  textInput.addEventListener("input", updateCharCount);

  clearBtn.addEventListener("click", () => {
    textInput.value = "";
    updateCharCount();
    textInput.focus();
  });

  // Language Tabs
  langTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      langTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      currentLang = tab.dataset.lang;
      
      // Update placeholder font hint
      if (currentLang === "gu") {
        textInput.placeholder = "ગુજરાતી લિપિમાં ટેક્સ્ટ દાખલ કરો...";
      } else {
        textInput.placeholder = "देवनागरी लिपि में पाठ लिखें...";
      }
      
      renderPresets(currentLang);
    });
  });

  // Format tokens into colorful pill badges
  function formatTokens(tokens, isClustered) {
    if (!tokens || tokens.length === 0) return "-";
    return tokens.map(t => {
      if (t === "<wb>") {
        return `<span style="background: rgba(255,255,255,0.06); padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; color: var(--text-muted); margin: 2px;">␣ wb</span>`;
      }
      const bg = isClustered ? "rgba(249, 115, 22, 0.2)" : "rgba(99, 102, 241, 0.2)";
      const color = isClustered ? "#fb923c" : "#818cf8";
      return `<span style="background: ${bg}; color: ${color}; padding: 3px 7px; border-radius: 4px; margin: 2px; display: inline-block; font-weight: 600;">${t}</span>`;
    }).join(" ");
  }

  // Live G2P Inspection
  async function inspectG2P() {
    const text = textInput.value.trim();
    if (!text) return;

    try {
      previewG2pBtn.disabled = true;
      previewG2pBtn.textContent = "🔍 Analyzing...";

      const res = await fetch("/api/g2p/convert", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, lang: currentLang })
      });
      const data = await res.json();

      if (data.status === "success") {
        g2pCard.style.display = "block";
        reductionBadge.textContent = `-${data.reduction_pct}% Token Reduction`;
        baseCount.textContent = `${data.baseline_count} tokens`;
        clustCount.textContent = `${data.clustered_count} tokens`;
        
        baseTokensDiv.innerHTML = formatTokens(data.baseline_tokens, false);
        clustTokensDiv.innerHTML = formatTokens(data.clustered_tokens, true);
      }
    } catch (err) {
      console.error("G2P inspection error", err);
    } finally {
      previewG2pBtn.disabled = false;
      previewG2pBtn.textContent = "🔍 Inspect G2P Tokens";
    }
  }

  previewG2pBtn.addEventListener("click", inspectG2P);

  // Live Synthesis
  async function synthesizeSpeech() {
    const text = textInput.value.trim();
    if (!text) {
      alert("Please enter or select some text to synthesize.");
      return;
    }

    try {
      synthesizeBtn.disabled = true;
      synthesizeBtn.textContent = "⏳ Synthesizing on GPU...";
      
      // Also trigger G2P inspect
      inspectG2P();

      const res = await fetch("/api/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, lang: currentLang, model_type: "both" })
      });
      const data = await res.json();

      if (data.status === "success") {
        const r = data.result;
        resultsArea.style.display = "block";

        // Baseline output
        if (r.baseline && r.baseline.success) {
          audioBase.src = r.baseline.audio_url + "?t=" + Date.now();
          audioBase.load();
          metaBase.innerHTML = `<span style="color: var(--color-success)">✓ Ready</span> • Latency: <strong>${r.baseline.latency_ms} ms</strong> • RTF: ${r.baseline.rtf}x • Duration: ${r.baseline.duration_sec}s`;
          downloadBase.href = r.baseline.audio_url;
          downloadBase.style.display = "inline-flex";
        } else {
          metaBase.innerHTML = `<span style="color: var(--color-danger)">✗ Failed: ${r.baseline?.error || 'Unknown error'}</span>`;
          downloadBase.style.display = "none";
        }

        // Clustered output
        if (r.clustered && r.clustered.success) {
          audioClust.src = r.clustered.audio_url + "?t=" + Date.now();
          audioClust.load();
          metaClust.innerHTML = `<span style="color: var(--color-success)">✓ Ready</span> • Latency: <strong>${r.clustered.latency_ms} ms</strong> • RTF: ${r.clustered.rtf}x • Duration: ${r.clustered.duration_sec}s`;
          downloadClust.href = r.clustered.audio_url;
          downloadClust.style.display = "inline-flex";
        } else {
          metaClust.innerHTML = `<span style="color: var(--color-danger)">✗ Failed: ${r.clustered?.error || 'Unknown error'}</span>`;
          downloadClust.style.display = "none";
        }

        // Auto-play Baseline first
        audioBase.play().catch(() => {});

        // Scroll smoothly to audio area
        resultsArea.scrollIntoView({ behavior: "smooth", block: "nearest" });

      } else {
        alert("Synthesis error: " + (data.message || "Failed"));
      }
    } catch (err) {
      console.error("Live synthesis error", err);
      alert("Network error: Could not complete GPU synthesis.");
    } finally {
      synthesizeBtn.disabled = false;
      synthesizeBtn.textContent = "⚡ Synthesize Both Models (GPU)";
    }
  }

  synthesizeBtn.addEventListener("click", synthesizeSpeech);

  // Play both in sequence
  playSeqBtn.addEventListener("click", () => {
    audioBase.currentTime = 0;
    audioBase.play();
    audioBase.onended = () => {
      setTimeout(() => {
        audioClust.currentTime = 0;
        audioClust.play();
      }, 300);
    };
  });

  // Keyboard shortcut Ctrl + Enter to synthesize
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") {
      e.preventDefault();
      synthesizeSpeech();
    }
  });

  // Initial load
  loadPresets();
});
