/**
 * Interactive Evaluation Page Logic (tts_app)
 * Handles blinded trial loading, dual audio playback, rating, keyboard shortcuts, and API submission.
 */

document.addEventListener("DOMContentLoaded", () => {
  const urlParams = new URLSearchParams(window.location.search);
  const sessionId = urlParams.get("session_id") || sessionStorage.getItem("tts_session_id");

  if (!sessionId) {
    alert("No active session found. Redirecting to home configuration.");
    window.location.href = "/";
    return;
  }

  let currentTrialIndex = 1;
  let totalTrials = 25;
  let trialStartTime = Date.now();

  // Current trial state
  let currentRating = {
    mos_a: null,
    mos_b: null,
    preference: null,
  };

  // DOM Elements
  const langBadge = document.getElementById("lang-badge");
  const setBadge = document.getElementById("set-badge");
  const curTrialNum = document.getElementById("cur-trial-num");
  const totalTrialNum = document.getElementById("total-trial-num");
  const progressFill = document.getElementById("progress-fill");
  const sentenceText = document.getElementById("sentence-text");
  
  const audioA = document.getElementById("audio-player-a");
  const audioB = document.getElementById("audio-player-b");
  const labelScoreA = document.getElementById("label-score-a");
  const labelScoreB = document.getElementById("label-score-b");
  
  const prevBtn = document.getElementById("prev-btn");
  const submitNextBtn = document.getElementById("submit-next-btn");
  const validationHint = document.getElementById("validation-hint");

  // Prevent overlapping audio playback
  audioA.addEventListener("play", () => {
    if (!audioB.paused) audioB.pause();
  });
  audioB.addEventListener("play", () => {
    if (!audioA.paused) audioA.pause();
  });

  // Rating buttons setup (Sample A)
  document.querySelectorAll("#scale-grid-a .rating-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#scale-grid-a .rating-btn").forEach(b => b.classList.remove("selected"));
      btn.classList.add("selected");
      currentRating.mos_a = parseFloat(btn.dataset.val);
      labelScoreA.textContent = `${currentRating.mos_a.toFixed(1)} / 5.0`;
      validateState();
    });
  });

  // Rating buttons setup (Sample B)
  document.querySelectorAll("#scale-grid-b .rating-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#scale-grid-b .rating-btn").forEach(b => b.classList.remove("selected"));
      btn.classList.add("selected");
      currentRating.mos_b = parseFloat(btn.dataset.val);
      labelScoreB.textContent = `${currentRating.mos_b.toFixed(1)} / 5.0`;
      validateState();
    });
  });

  // Preference buttons setup
  document.querySelectorAll(".pref-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".pref-btn").forEach(b => b.classList.remove("selected"));
      btn.classList.add("selected");
      currentRating.preference = btn.dataset.pref;
      validateState();
    });
  });

  function resetRatingUI() {
    currentRating = { mos_a: null, mos_b: null, preference: null };
    document.querySelectorAll(".rating-btn").forEach(b => b.classList.remove("selected"));
    document.querySelectorAll(".pref-btn").forEach(b => b.classList.remove("selected"));
    labelScoreA.textContent = "Select score";
    labelScoreB.textContent = "Select score";
    validationHint.style.display = "none";
    trialStartTime = Date.now();
  }

  function validateState() {
    const isValid = (
      currentRating.mos_a !== null &&
      currentRating.mos_b !== null &&
      currentRating.preference !== null
    );
    if (isValid) {
      validationHint.style.display = "none";
    }
    return isValid;
  }

  // Fetch and display a trial
  async function loadTrial(index) {
    try {
      submitNextBtn.disabled = true;
      sentenceText.textContent = "Loading sentence...";
      
      const res = await fetch(`/api/trial/${sessionId}/${index}`);
      const data = await res.json();

      if (data.status !== "success") {
        alert(data.message || "Failed to load trial");
        return;
      }

      currentTrialIndex = data.trial_index;
      totalTrials = data.total_trials;

      // Update UI Header
      const langFlags = { hi: "🇮🇳 Hindi", mr: "🚩 Marathi", gu: "🦁 Gujarati" };
      langBadge.textContent = langFlags[data.language] || data.lang_name;
      setBadge.textContent = data.eval_set === "held_out" ? "Held-Out Test" : "Unseen Sentence";
      curTrialNum.textContent = currentTrialIndex;
      totalTrialNum.textContent = totalTrials;
      
      const pct = Math.round(((currentTrialIndex - 1) / totalTrials) * 100);
      progressFill.style.width = `${pct}%`;

      // Update Sentence Text
      sentenceText.textContent = data.text;

      // Update Audio Sources
      audioA.src = data.audio_a_url;
      audioB.src = data.audio_b_url;
      audioA.load();
      audioB.load();

      // Reset rating state
      resetRatingUI();

      // If already rated previously, restore values
      if (data.existing_rating) {
        const er = data.existing_rating;
        const btnA = document.querySelector(`#scale-grid-a .rating-btn[data-val="${er.mos_a.toFixed(1)}"]`);
        if (btnA) btnA.click();

        const btnB = document.querySelector(`#scale-grid-b .rating-btn[data-val="${er.mos_b.toFixed(1)}"]`);
        if (btnB) btnB.click();

        const pBtn = document.querySelector(`.pref-btn[data-pref="${er.preference}"]`);
        if (pBtn) pBtn.click();
      }

      // Update Navigation
      prevBtn.disabled = currentTrialIndex <= 1;
      submitNextBtn.disabled = false;
      submitNextBtn.textContent = (currentTrialIndex === totalTrials) ? "🎉 Finish & View Results" : "Submit & Next ➡️";

    } catch (err) {
      console.error("Error loading trial", err);
      alert("Network error: Failed to fetch trial audio.");
    }
  }

  // Handle Submit & Next
  async function submitTrial() {
    if (!validateState()) {
      validationHint.style.display = "inline";
      return;
    }

    const durationSec = ((Date.now() - trialStartTime) / 1000).toFixed(1);
    submitNextBtn.disabled = true;
    submitNextBtn.textContent = "💾 Saving...";

    const payload = {
      session_id: sessionId,
      trial_index: currentTrialIndex,
      mos_a: currentRating.mos_a,
      mos_b: currentRating.mos_b,
      preference: currentRating.preference,
      duration_sec: parseFloat(durationSec),
    };

    try {
      const res = await fetch("/api/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (data.status === "success") {
        if (data.is_finished || !data.next_trial) {
          // Session complete!
          window.location.href = `/results?completed=1&session_id=${sessionId}`;
        } else {
          loadTrial(data.next_trial);
        }
      } else {
        alert(data.message || "Could not save rating");
        submitNextBtn.disabled = false;
        submitNextBtn.textContent = "Submit & Next ➡️";
      }
    } catch (err) {
      console.error("Error submitting rating", err);
      alert("Network error: Could not submit rating.");
      submitNextBtn.disabled = false;
      submitNextBtn.textContent = "Submit & Next ➡️";
    }
  }

  submitNextBtn.addEventListener("click", submitTrial);

  prevBtn.addEventListener("click", () => {
    if (currentTrialIndex > 1) {
      loadTrial(currentTrialIndex - 1);
    }
  });

  // Keyboard Shortcuts
  document.addEventListener("keydown", (e) => {
    // Only trigger if not typing in an input
    if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;

    if (e.key === "1") {
      e.preventDefault();
      audioA.currentTime = 0;
      audioA.play();
    } else if (e.key === "2") {
      e.preventDefault();
      audioB.currentTime = 0;
      audioB.play();
    } else if (e.key === "Enter") {
      e.preventDefault();
      submitTrial();
    }
  });

  // Initial load
  loadTrial(1);
});
