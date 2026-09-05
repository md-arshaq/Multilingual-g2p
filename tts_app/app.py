#!/usr/bin/env python3
"""
Multilingual TTS Human Evaluation Web Application.

Enables blind AB testing of Baseline (57 phonemes) vs Clustered (39 clusters)
VITS models across Hindi, Marathi, and Gujarati.

Features:
- Subfolder-contained web app architecture (`tts_app/`)
- Dynamic evaluator session creation & resumption
- Language selection (Hindi, Marathi, Gujarati, All)
- Customizable sample count (10 to max available)
- Deterministic AB randomization per evaluator
- Dual audio playback, tactile 1-5 MOS rating, pairwise preference
- Unblinded real-time results dashboard with statistical charts & CSV export
"""

import csv
import hashlib
import json
import os
import random
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from flask import (
    Flask, jsonify, render_template, request,
    send_from_directory, redirect, url_for, Response
)

# Base directories
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
TEMPLATE_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "human_eval.db"

# Import live synthesizer engine
try:
    from synthesizer import get_synthesizer, PRESETS, GENERATED_DIR
except ImportError:
    from tts_app.synthesizer import get_synthesizer, PRESETS, GENERATED_DIR

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static"
)
app.config["SECRET_KEY"] = "multilingual-tts-human-eval-secret-2026"

# Language and sample directory configurations
LANG_CONFIGS = {
    "hi": {
        "name": "Hindi",
        "script": "Devanagari",
        "metadata_csv": PROJECT_ROOT / "samples" / "tts_hindi_female" / "inference_metadata.csv",
        "audio_dir": PROJECT_ROOT / "samples" / "tts_hindi_female",
    },
    "mr": {
        "name": "Marathi",
        "script": "Devanagari",
        "metadata_csv": PROJECT_ROOT / "samples" / "tts_marathi_female" / "inference_metadata.csv",
        "audio_dir": PROJECT_ROOT / "samples" / "tts_marathi_female",
    },
    "gu": {
        "name": "Gujarati",
        "script": "Gujarati",
        "metadata_csv": PROJECT_ROOT / "samples" / "tts_gujarati_female" / "inference_metadata.csv",
        "audio_dir": PROJECT_ROOT / "samples" / "tts_gujarati_female",
    },
}


# ── Database Setup ─────────────────────────────────────────────────────────────

def get_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluator_id INTEGER NOT NULL REFERENCES evaluators(id),
        language TEXT NOT NULL,
        eval_set TEXT NOT NULL,
        requested_count INTEGER NOT NULL,
        trials_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL REFERENCES sessions(id),
        trial_index INTEGER NOT NULL,
        language TEXT NOT NULL,
        eval_set TEXT NOT NULL,
        sample_index INTEGER NOT NULL,
        sample_id TEXT,
        text TEXT NOT NULL,
        a_is_baseline BOOLEAN NOT NULL,
        mos_a REAL NOT NULL,
        mos_b REAL NOT NULL,
        preference TEXT NOT NULL,
        duration_sec REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(session_id, trial_index)
    );
    """)
    
    conn.commit()
    conn.close()


# ── Metadata & Sample Loader ──────────────────────────────────────────────────

def load_all_sample_pairs():
    """Load all synthesized sample pairs across all languages from inference metadata."""
    sample_pool = []
    for lang_code, cfg in LANG_CONFIGS.items():
        csv_path = cfg["metadata_csv"]
        if not csv_path.exists():
            continue
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                baseline_wav = row.get("baseline_wav", "")
                clustered_wav = row.get("clustered_wav", "")
                eval_set = row.get("set", "held_out")
                
                if not baseline_wav or not clustered_wav:
                    continue
                
                b_path = cfg["audio_dir"] / eval_set / baseline_wav
                c_path = cfg["audio_dir"] / eval_set / clustered_wav
                
                if b_path.exists() and c_path.exists():
                    sample_pool.append({
                        "language": lang_code,
                        "lang_name": cfg["name"],
                        "eval_set": eval_set,
                        "sample_index": int(row.get("index", 0)),
                        "sample_id": row.get("sample_id", ""),
                        "text": row.get("text", ""),
                        "baseline_wav": baseline_wav,
                        "clustered_wav": clustered_wav,
                    })
    return sample_pool


# ── Deterministic AB Randomization ────────────────────────────────────────────

def create_randomized_trials(evaluator_name, language_choice, eval_set_choice, sample_count):
    """
    Select and randomize trial order and AB assignment deterministically
    for a given evaluator session.
    """
    all_samples = load_all_sample_pairs()
    
    # Filter by language
    if language_choice != "all":
        candidates = [s for s in all_samples if s["language"] == language_choice]
    else:
        candidates = all_samples
        
    # Filter by set
    if eval_set_choice in ["held_out", "unseen"]:
        candidates = [s for s in candidates if s["eval_set"] == eval_set_choice]
        
    if not candidates:
        return []
    
    seed_str = f"{evaluator_name}_{language_choice}_{eval_set_choice}_{datetime.now().isoformat()}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    
    # Stratify held_out vs unseen if both requested
    if eval_set_choice == "both":
        held_out = [s for s in candidates if s["eval_set"] == "held_out"]
        unseen = [s for s in candidates if s["eval_set"] == "unseen"]
        
        rng.shuffle(held_out)
        rng.shuffle(unseen)
        
        total_avail = len(held_out) + len(unseen)
        target_count = min(sample_count, total_avail)
        
        n_held = int(round(target_count * (len(held_out) / total_avail)))
        n_unseen = target_count - n_held
        
        n_held = min(n_held, len(held_out))
        n_unseen = min(target_count - n_held, len(unseen))
        n_held = target_count - n_unseen
        
        selected = held_out[:n_held] + unseen[:n_unseen]
    else:
        rng.shuffle(candidates)
        selected = candidates[:min(sample_count, len(candidates))]
        
    rng.shuffle(selected)
    
    trials = []
    for i, s in enumerate(selected):
        a_is_baseline = rng.choice([True, False])
        
        trials.append({
            "trial_index": i + 1,
            "language": s["language"],
            "lang_name": s["lang_name"],
            "eval_set": s["eval_set"],
            "sample_index": s["sample_index"],
            "sample_id": s["sample_id"],
            "text": s["text"],
            "a_is_baseline": a_is_baseline,
            "audio_a_file": s["baseline_wav"] if a_is_baseline else s["clustered_wav"],
            "audio_b_file": s["clustered_wav"] if a_is_baseline else s["baseline_wav"],
        })
        
    return trials


# ── Audio Serving Route ───────────────────────────────────────────────────────

@app.route("/audio/<lang>/<eval_set>/<filename>")
def serve_audio(lang, eval_set, filename):
    """Serve audio WAV files dynamically."""
    if lang not in LANG_CONFIGS:
        return "Invalid language", 404
    
    dir_path = LANG_CONFIGS[lang]["audio_dir"] / eval_set
    if not dir_path.exists():
        return "Directory not found", 404
        
    return send_from_directory(str(dir_path), filename, mimetype="audio/wav")


# ── HTML View Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/evaluate")
def evaluate_page():
    return render_template("evaluate.html")


@app.route("/results")
def results_page():
    return render_template("results.html")


@app.route("/demo")
def demo_page():
    """Live TTS synthesis playground."""
    return render_template("demo.html")


@app.route("/audio/generated/<filename>")
def serve_generated_audio(filename):
    """Serve dynamically synthesized WAV audio files."""
    return send_from_directory(str(GENERATED_DIR), filename, mimetype="audio/wav")


# ── Live Synthesis Playground APIs ───────────────────────────────────────────

@app.route("/api/presets")
def api_presets():
    """Return preloaded sample sentences categorized by domain."""
    lang = request.args.get("lang", "all")
    if lang in PRESETS:
        return jsonify({"status": "success", "presets": PRESETS[lang]})
    return jsonify({"status": "success", "presets": PRESETS})


@app.route("/api/g2p/convert", methods=["POST"])
def api_g2p_convert():
    """Real-time G2P phoneme breakdown preview."""
    data = request.json or {}
    text = (data.get("text") or "").strip()
    lang = data.get("lang", "hi")
    
    if not text:
        return jsonify({"status": "error", "message": "Text is required"}), 400
        
    try:
        synth = get_synthesizer()
        b_seq, c_seq, b_toks, c_toks = synth.text_to_tokens(text, lang)
        
        reduction = 0.0
        if len(b_toks) > 0:
            reduction = round(((len(b_toks) - len(c_toks)) / len(b_toks)) * 100, 1)
            
        return jsonify({
            "status": "success",
            "text": text,
            "lang": lang,
            "baseline_seq": b_seq,
            "clustered_seq": c_seq,
            "baseline_tokens": b_toks,
            "clustered_tokens": c_toks,
            "baseline_count": len(b_toks),
            "clustered_count": len(c_toks),
            "reduction_pct": reduction,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/synthesize", methods=["POST"])
def api_synthesize():
    """Live GPU synthesis for arbitrary text."""
    data = request.json or {}
    text = (data.get("text") or "").strip()
    lang = data.get("lang", "hi")
    model_type = data.get("model_type", "both")
    
    if not text:
        return jsonify({"status": "error", "message": "Text is required"}), 400
        
    try:
        synth = get_synthesizer()
        result = synth.synthesize(text, lang, model_type)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.route("/api/available_counts")
def api_available_counts():
    """Return total available pair counts per language and split."""
    samples = load_all_sample_pairs()
    counts = {
        "all": {"held_out": 0, "unseen": 0, "total": 0},
        "hi": {"held_out": 0, "unseen": 0, "total": 0},
        "mr": {"held_out": 0, "unseen": 0, "total": 0},
        "gu": {"held_out": 0, "unseen": 0, "total": 0},
    }
    
    for s in samples:
        lang = s["language"]
        eset = s["eval_set"]
        if lang in counts:
            counts[lang][eset] += 1
            counts[lang]["total"] += 1
            counts["all"][eset] += 1
            counts["all"]["total"] += 1
            
    return jsonify({"status": "success", "counts": counts})


@app.route("/api/evaluators")
def api_evaluators():
    """List existing evaluators and their recent sessions."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT e.id as evaluator_id, e.name, s.id as session_id, s.language, s.eval_set, 
           s.requested_count, s.completed_at,
           COUNT(r.id) as completed_trials
    FROM evaluators e
    JOIN sessions s ON e.id = s.evaluator_id
    LEFT JOIN ratings r ON s.id = r.session_id
    GROUP BY s.id
    ORDER BY s.created_at DESC
    LIMIT 20;
    """)
    
    rows = cursor.fetchall()
    evaluators = []
    for row in rows:
        evaluators.append({
            "evaluator_id": row["evaluator_id"],
            "name": row["name"],
            "session_id": row["session_id"],
            "language": row["language"],
            "eval_set": row["eval_set"],
            "requested_count": row["requested_count"],
            "completed_trials": row["completed_trials"],
            "is_complete": row["completed_at"] is not None or row["completed_trials"] >= row["requested_count"],
        })
        
    conn.close()
    return jsonify({"status": "success", "sessions": evaluators})


@app.route("/api/session", methods=["POST"])
def api_create_session():
    """Create or resume an evaluation session."""
    data = request.json or {}
    name = (data.get("name") or "").strip()
    language = data.get("language", "all")
    eval_set = data.get("eval_set", "both")
    requested_count = int(data.get("count", 25))
    resume_session_id = data.get("resume_session_id")
    
    if not name:
        return jsonify({"status": "error", "message": "Evaluator name is required"}), 400
        
    conn = get_db()
    cursor = conn.cursor()
    
    if resume_session_id:
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (resume_session_id,))
        session_row = cursor.fetchone()
        if session_row:
            cursor.execute("SELECT COUNT(*) as cnt FROM ratings WHERE session_id = ?", (resume_session_id,))
            completed = cursor.fetchone()["cnt"]
            conn.close()
            return jsonify({
                "status": "success",
                "session_id": session_row["id"],
                "name": name,
                "language": session_row["language"],
                "eval_set": session_row["eval_set"],
                "total_trials": session_row["requested_count"],
                "completed_trials": completed,
                "resumed": True,
            })
            
    cursor.execute("SELECT id FROM evaluators WHERE name = ?", (name,))
    eval_row = cursor.fetchone()
    if eval_row:
        evaluator_id = eval_row["id"]
    else:
        cursor.execute("INSERT INTO evaluators (name) VALUES (?)", (name,))
        evaluator_id = cursor.lastrowid
        
    trials = create_randomized_trials(name, language, eval_set, requested_count)
    if not trials:
        conn.close()
        return jsonify({"status": "error", "message": "No audio pairs match this criteria"}), 400
        
    actual_count = len(trials)
    trials_json = json.dumps(trials)
    
    cursor.execute("""
    INSERT INTO sessions (evaluator_id, language, eval_set, requested_count, trials_json)
    VALUES (?, ?, ?, ?, ?)
    """, (evaluator_id, language, eval_set, actual_count, trials_json))
    
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({
        "status": "success",
        "session_id": session_id,
        "name": name,
        "language": language,
        "eval_set": eval_set,
        "total_trials": actual_count,
        "completed_trials": 0,
        "resumed": False,
    })


@app.route("/api/trial/<int:session_id>/<int:trial_index>")
def api_get_trial(session_id, trial_index):
    """Retrieve trial details for evaluation."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session_row = cursor.fetchone()
    if not session_row:
        conn.close()
        return jsonify({"status": "error", "message": "Session not found"}), 404
        
    trials = json.loads(session_row["trials_json"])
    if trial_index < 1 or trial_index > len(trials):
        conn.close()
        return jsonify({"status": "error", "message": "Invalid trial index"}), 404
        
    trial_data = trials[trial_index - 1]
    
    cursor.execute("""
    SELECT mos_a, mos_b, preference, duration_sec FROM ratings 
    WHERE session_id = ? AND trial_index = ?
    """, (session_id, trial_index))
    existing_rating = cursor.fetchone()
    
    cursor.execute("SELECT COUNT(*) as cnt FROM ratings WHERE session_id = ?", (session_id,))
    completed_cnt = cursor.fetchone()["cnt"]
    
    conn.close()
    
    response = {
        "status": "success",
        "session_id": session_id,
        "trial_index": trial_index,
        "total_trials": len(trials),
        "completed_trials": completed_cnt,
        "language": trial_data["language"],
        "lang_name": trial_data["lang_name"],
        "eval_set": trial_data["eval_set"],
        "text": trial_data["text"],
        "audio_a_url": f"/audio/{trial_data['language']}/{trial_data['eval_set']}/{trial_data['audio_a_file']}",
        "audio_b_url": f"/audio/{trial_data['language']}/{trial_data['eval_set']}/{trial_data['audio_b_file']}",
        "existing_rating": {
            "mos_a": existing_rating["mos_a"],
            "mos_b": existing_rating["mos_b"],
            "preference": existing_rating["preference"],
        } if existing_rating else None
    }
    return jsonify(response)


@app.route("/api/submit", methods=["POST"])
def api_submit_rating():
    """Submit rating for a trial."""
    data = request.json or {}
    session_id = data.get("session_id")
    trial_index = data.get("trial_index")
    mos_a = float(data.get("mos_a", 0))
    mos_b = float(data.get("mos_b", 0))
    preference = data.get("preference", "none")
    duration_sec = float(data.get("duration_sec", 0.0))
    
    if not (1.0 <= mos_a <= 5.0 and 1.0 <= mos_b <= 5.0):
        return jsonify({"status": "error", "message": "MOS ratings must be between 1.0 and 5.0"}), 400
        
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session_row = cursor.fetchone()
    if not session_row:
        conn.close()
        return jsonify({"status": "error", "message": "Session not found"}), 404
        
    trials = json.loads(session_row["trials_json"])
    if trial_index < 1 or trial_index > len(trials):
        conn.close()
        return jsonify({"status": "error", "message": "Invalid trial index"}), 400
        
    trial_data = trials[trial_index - 1]
    a_is_baseline = trial_data["a_is_baseline"]
    
    cursor.execute("""
    INSERT INTO ratings (
        session_id, trial_index, language, eval_set, sample_index, 
        sample_id, text, a_is_baseline, mos_a, mos_b, preference, duration_sec
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(session_id, trial_index) DO UPDATE SET
        mos_a = excluded.mos_a,
        mos_b = excluded.mos_b,
        preference = excluded.preference,
        duration_sec = excluded.duration_sec,
        created_at = CURRENT_TIMESTAMP
    """, (
        session_id, trial_index, trial_data["language"], trial_data["eval_set"],
        trial_data["sample_index"], trial_data["sample_id"], trial_data["text"],
        a_is_baseline, mos_a, mos_b, preference, duration_sec
    ))
    
    cursor.execute("SELECT COUNT(*) as cnt FROM ratings WHERE session_id = ?", (session_id,))
    completed_cnt = cursor.fetchone()["cnt"]
    
    if completed_cnt >= len(trials):
        cursor.execute("UPDATE sessions SET completed_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
        
    conn.commit()
    conn.close()
    
    return jsonify({
        "status": "success",
        "completed_trials": completed_cnt,
        "total_trials": len(trials),
        "is_finished": completed_cnt >= len(trials),
        "next_trial": trial_index + 1 if trial_index < len(trials) else None,
    })


@app.route("/api/results")
def api_get_results():
    """Unblind ratings to calculate human MOS and preference distributions."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT r.*, e.name as evaluator_name
    FROM ratings r
    JOIN sessions s ON r.session_id = s.id
    JOIN evaluators e ON s.evaluator_id = e.id
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return jsonify({
            "status": "success",
            "total_ratings": 0,
            "evaluators_count": 0,
            "evaluators": [],
            "by_language": {},
            "overall": {
                "baseline": {"n": 0, "mean": 0.0, "std": 0.0, "ci_lo": 0.0, "ci_hi": 0.0},
                "clustered": {"n": 0, "mean": 0.0, "std": 0.0, "ci_lo": 0.0, "ci_hi": 0.0},
            },
            "preferences": {"baseline": 0, "clustered": 0, "none": 0},
        })
        
    by_lang = {
        "hi": {"baseline": [], "clustered": [], "pref": {"baseline": 0, "clustered": 0, "none": 0}},
        "mr": {"baseline": [], "clustered": [], "pref": {"baseline": 0, "clustered": 0, "none": 0}},
        "gu": {"baseline": [], "clustered": [], "pref": {"baseline": 0, "clustered": 0, "none": 0}},
    }
    overall = {"baseline": [], "clustered": []}
    pref_overall = {"baseline": 0, "clustered": 0, "none": 0}
    evaluators_set = set()
    
    for r in rows:
        lang = r["language"]
        evaluators_set.add(r["evaluator_name"])
        
        if r["a_is_baseline"]:
            mos_base = r["mos_a"]
            mos_clust = r["mos_b"]
            if r["preference"] == "A":
                winner = "baseline"
            elif r["preference"] == "B":
                winner = "clustered"
            else:
                winner = "none"
        else:
            mos_base = r["mos_b"]
            mos_clust = r["mos_a"]
            if r["preference"] == "A":
                winner = "clustered"
            elif r["preference"] == "B":
                winner = "baseline"
            else:
                winner = "none"
                
        if lang in by_lang:
            by_lang[lang]["baseline"].append(mos_base)
            by_lang[lang]["clustered"].append(mos_clust)
            by_lang[lang]["pref"][winner] += 1
            
        overall["baseline"].append(mos_base)
        overall["clustered"].append(mos_clust)
        pref_overall[winner] += 1
        
    def get_summary(scores):
        if not scores:
            return {"n": 0, "mean": 0.0, "std": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
        n = len(scores)
        mean = sum(scores) / n
        std = (sum((x - mean) ** 2 for x in scores) / (n - 1)) ** 0.5 if n > 1 else 0.0
        se = std / (n ** 0.5) if n > 0 else 0.0
        return {
            "n": n,
            "mean": round(mean, 3),
            "std": round(std, 3),
            "ci_lo": round(mean - 1.96 * se, 3),
            "ci_hi": round(mean + 1.96 * se, 3),
        }
        
    summary_by_lang = {}
    for lang, data in by_lang.items():
        summary_by_lang[lang] = {
            "name": LANG_CONFIGS[lang]["name"],
            "baseline": get_summary(data["baseline"]),
            "clustered": get_summary(data["clustered"]),
            "preferences": data["pref"],
        }
        
    return jsonify({
        "status": "success",
        "total_ratings": len(rows),
        "evaluators_count": len(evaluators_set),
        "evaluators": list(evaluators_set),
        "by_language": summary_by_lang,
        "overall": {
            "baseline": get_summary(overall["baseline"]),
            "clustered": get_summary(overall["clustered"]),
        },
        "preferences": pref_overall,
    })


@app.route("/api/export/csv")
def api_export_csv():
    """Export all unblinded ratings as a downloadable CSV."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT r.*, e.name as evaluator_name, s.created_at as session_date
    FROM ratings r
    JOIN sessions s ON r.session_id = s.id
    JOIN evaluators e ON s.evaluator_id = e.id
    ORDER BY r.created_at ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        "rating_id", "session_id", "evaluator", "language", "eval_set",
        "sample_index", "sample_id", "text", "a_is_baseline",
        "mos_a", "mos_b", "mos_baseline", "mos_clustered",
        "raw_preference", "unblinded_preference", "duration_sec", "created_at"
    ])
    
    for r in rows:
        if r["a_is_baseline"]:
            mos_base = r["mos_a"]
            mos_clust = r["mos_b"]
            if r["preference"] == "A":
                unblind_pref = "Baseline"
            elif r["preference"] == "B":
                unblind_pref = "Clustered"
            else:
                unblind_pref = "No Preference"
        else:
            mos_base = r["mos_b"]
            mos_clust = r["mos_a"]
            if r["preference"] == "A":
                unblind_pref = "Clustered"
            elif r["preference"] == "B":
                unblind_pref = "Baseline"
            else:
                unblind_pref = "No Preference"
                
        writer.writerow([
            r["id"], r["session_id"], r["evaluator_name"], r["language"], r["eval_set"],
            r["sample_index"], r["sample_id"], r["text"], r["a_is_baseline"],
            r["mos_a"], r["mos_b"], mos_base, mos_clust,
            r["preference"], unblind_pref, r["duration_sec"], r["created_at"]
        ])
        
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=human_eval_ratings.csv"}
    )


if __name__ == "__main__":
    init_db()
    print("=" * 60)
    print("  MULTILINGUAL TTS APP & LIVE PLAYGROUND (tts_app/)")
    print("=" * 60)
    print(f"  Database: {DB_PATH}")
    print("  Available Languages: Hindi, Marathi, Gujarati")
    print("  Server Address: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
