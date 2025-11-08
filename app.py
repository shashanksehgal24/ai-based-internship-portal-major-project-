# app.py
from flask import Flask, request, jsonify, send_from_directory
import os, joblib, traceback, re
import pandas as pd

APP_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(APP_DIR, "static")
BUNDLE_PATH = os.path.join(APP_DIR, "internship_company_recommender.joblib")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

# helper to load bundle once
_bundle = None
def load_bundle():
    global _bundle
    if _bundle is not None:
        return _bundle
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(f"Bundle not found at {BUNDLE_PATH}. Place internship_company_recommender.joblib here.")
    _bundle = joblib.load(BUNDLE_PATH)
    return _bundle

def parse_experience(x):
    try:
        if x is None: return 0
        if isinstance(x, (int,float)): return int(x)
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else 0
    except:
        return 0

def safe_label_encode(le, value):
    try:
        if value in list(le.classes_):
            return int(le.transform([value])[0])
    except Exception:
        pass
    return 0

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/results")
def results_page():
    return send_from_directory(STATIC_DIR, "results.html")

@app.route("/insights")
def insights_page():
    return send_from_directory(STATIC_DIR, "insights.html")

@app.route("/history")
def history_page():
    return send_from_directory(STATIC_DIR, "history.html")

# API: POST JSON -> prediction
# Expected JSON fields: Age, Gender, Location, CGPA, Technical skills, Work experience, Company location, Rating by company, Internship status
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        bundle = load_bundle()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"ok": False, "error": "Invalid or empty JSON body"}), 400

    try:
        student = {
            "Age": int(data.get("Age", 21)),
            "Gender": data.get("Gender", "Male"),
            "Location": data.get("Location", "Delhi"),
            "CGPA": float(data.get("CGPA", 8.0)),
            "Technical skills": data.get("Technical skills", ""),
            "Work experience": data.get("Work experience", "0"),
            "Company location": data.get("Company location", "Bangalore"),
            "Rating by company": int(data.get("Rating by company", 0)),
            "Internship status": data.get("Internship status", "NO")
        }

        vec = bundle["vectorizer"].transform([student["Technical skills"]])
        skill_cols = [f"skill_{s}" for s in bundle["vectorizer"].get_feature_names_out()]
        skills_df = pd.DataFrame(vec.toarray(), columns=skill_cols)

        row = {
            "Age": student["Age"],
            "Gender_enc": safe_label_encode(bundle["le_gender"], student["Gender"]),
            "CGPA": student["CGPA"],
            "Location_enc": safe_label_encode(bundle["le_location"], student["Location"]),
            "Company_location_enc": safe_label_encode(bundle["le_company_loc"], student["Company location"]),
            "Work experience months": parse_experience(student["Work experience"]),
            "Internship status bin": 1 if str(student["Internship status"]).upper() == "YES" else 0,
            "Rating by company": student["Rating by company"]
        }

        X_new = pd.DataFrame([row])
        X_new = pd.concat([X_new.reset_index(drop=True), skills_df], axis=1)

        # add missing features
        for c in bundle["feature_columns"]:
            if c not in X_new.columns:
                X_new[c] = 0
        X_new = X_new[bundle["feature_columns"]]

        model = bundle["model"]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0]
            # classes are encoded 0..n-1; inverse transform to names
            classes = bundle["le_company"].inverse_transform(list(range(len(proba))))
            pairs = sorted(list(zip(classes, proba)), key=lambda x: x[1], reverse=True)[:5]
            top3 = pairs[:3]
        else:
            pred = model.predict(X_new)[0]
            comp = bundle["le_company"].inverse_transform([pred])[0]
            top3 = [(comp, 1.0)]

        # matched skills heuristic (optional)
        matched = {}
        try:
            # if dataset present, try to read company required skills for better match
            csv_path = os.path.join(APP_DIR, "internshipdata.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for c,_ in top3:
                    reqs = df[df["Company name"]==c]["Skills required by company"].astype(str).tolist()
                    tokens = []
                    for r in reqs:
                        tokens += re.findall(r"\\w+", r.lower())
                    student_tokens = re.findall(r"\\w+", student["Technical skills"].lower())
                    matched[c] = list(set(tokens) & set(student_tokens))
            else:
                for c,_ in top3:
                    matched[c] = []
        except Exception:
            for c,_ in top3:
                matched[c] = []

        result = {
            "ok": True,
            "top3": [(c, float(p)) for c,p in top3],
            "matched": matched,
            "notes": "Returned top-3 companies (model probabilities)."
        }
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7860)
