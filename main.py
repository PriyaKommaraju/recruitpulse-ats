import os
import json
import time
import requests
import pdfplumber
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ---------------- APP SETUP ----------------

app = Flask(__name__, template_folder="templates")
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔑 ADD YOUR GEMINI API KEY
GEMINI_API_KEY = "AIzaSyDjf2cJS4p488XvX0J4ttbK3AM6rtCpQmI"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.5-flash-preview-09-2025:generateContent"
    f"?key={GEMINI_API_KEY}"
)

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF resume"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        resume_text = extract_text_from_pdf(path)

        if len(resume_text.strip()) < 300:
            return jsonify({"error": "Resume text too short or unreadable"}), 422

        # ---------- STABLE ATS SCORING ----------
        keyword_score, formatting_score, technical_score = stable_ats_scores(resume_text)

        ats_score = round(
            keyword_score * 0.45 +
            formatting_score * 0.20 +
            technical_score * 0.35
        )

        # ---------- AI FOR INSIGHTS ONLY ----------
        ai_data = analyze_with_gemini(resume_text)

        response = {
            "ats_score": ats_score,
            "keyword_score": keyword_score,
            "formatting_score": formatting_score,
            "technical_score": technical_score,
            **ai_data
        }

        os.remove(path)
        return jsonify(response)

    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        return jsonify({"error": str(e)}), 500


# ---------------- HELPERS ----------------

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


# ---------------- DETERMINISTIC ATS SCORING ----------------

def stable_ats_scores(resume_text):
    """
    Rule-based ATS scoring.
    Same resume → same score always.
    """

    text = resume_text.lower()

    # Keyword score
    core_keywords = [
        "python", "java", "sql", "projects", "experience",
        "internship", "skills", "github", "api"
    ]
    keyword_hits = sum(1 for k in core_keywords if k in text)
    keyword_score = min(90, 60 + keyword_hits * 4)

    # Formatting score
    required_sections = ["skills", "experience", "projects"]
    formatting_score = 85 if all(s in text for s in required_sections) else 70

    # Technical score
    technical_terms = [
        "framework", "database", "api", "cloud",
        "backend", "frontend", "sql"
    ]
    tech_hits = sum(1 for t in technical_terms if t in text)
    technical_score = min(90, 60 + tech_hits * 5)

    return keyword_score, formatting_score, technical_score


# ---------------- AI FOR TEXT OUTPUT ONLY ----------------

def analyze_with_gemini(resume_text):
    """
    AI is used ONLY for explanations.
    Scores are NOT taken from AI.
    Strengths are strictly limited to 3.
    """

    prompt = f"""
You are a senior recruiter.

STRICT RULES:
- Do NOT give numeric scores
- Keep output short, factual, and resume-based
- ATS Improvements → keywords, sections, formatting only
- Technical Improvements → skills, tools, project depth only
- Avoid assumptions and generic claims

OUTPUT VALID JSON ONLY:

{{
  "strengths": ["clear resume-based strength"],
  "ats_improvements": ["ATS-specific improvement"],
  "technical_improvements": ["technical improvement"],
  "recommended_job_roles": ["role name"],
  "overall_summary": "2 concise sentences"
}}

RESUME:
\"\"\"{resume_text}\"\"\"
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json"
        }
    }

    retries = 3
    delay = 2

    for _ in range(retries):
        response = requests.post(GEMINI_URL, json=payload, timeout=40)

        if response.status_code == 200:
            raw = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            data = json.loads(raw)

            # 🔒 HARD LIMIT: ALWAYS 3 STRENGTHS
            data["strengths"] = data.get("strengths", [])[:3]

            return data

        if response.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue

        response.raise_for_status()

    raise Exception("AI service unavailable")


# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
