# app.py -- robust Flask app for internship recommender
import os, re, traceback
from flask import Flask, request, render_template_string
APP_DIR = os.path.dirname(__file__)

# Template used when imports fail (shows missing modules + install command)
MISSING_TEMPLATE = """
<!doctype html>
<title>Missing modules</title>
<h2 style="color:darkred">Application cannot start — missing Python modules</h2>
<p>The server failed to import some required Python packages. Install them with the command shown below and restart the app.</p>
<pre style="background:#f6f6f6;padding:10px">{{ install_cmd }}</pre>
<p>Missing packages detected:</p>
<ul>
{% for pkg in missing %}
  <li>{{ pkg }}</li>
{% endfor %}
</ul>
<hr>
<p>Full import error:</p>
<pre style="background:#eee;padding:10px">{{ err }}</pre>
"""

# Try to import heavy deps; if something fails capture it and render a helpful page
missing = []
_import_error = None
try:
    import joblib
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
except Exception as e:
    _import_error = traceback.format_exc()
    # simple heuristic to list likely missing pip packages from the exception text
    text = str(e).lower()
    candidates = []
    for name, pkg in [("joblib","joblib"),("pandas","pandas"),("sklearn","scikit-learn"),("sklearn.preprocessing","scikit-learn"),("sklearn.ensemble","scikit-learn"),("flask","flask")]:
        if name in text or name.replace('.','/') in text:
            candidates.append(pkg)
    # always include basic list so user can run pip install easily
    missing = list(dict.fromkeys(candidates or ["flask","pandas","scikit-learn","joblib"]))
    
# Flask import must be attempted after we see whether it's missing; try now:
try:
    from flask import Flask, request, render_template, render_template_string, redirect, url_for
except Exception as e:
    if "flask" not in missing:
        missing.append("flask")
    if not _import_error:
        _import_error = traceback.format_exc()

app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-key"

# If imports failed, show a friendly page on all routes
if missing:
    @app.route("/", methods=["GET"])
    def missing_page():
        install_cmd = "python -m pip install " + " ".join(sorted(set(missing)))
        return render_template_string(MISSING_TEMPLATE, missing=missing, install_cmd=install_cmd, err=_import_error or "")
    # also provide a catch-all so every path shows same error
    @app.route("/<path:_any>")
    def missing_catchall(_any):
        install_cmd = "python -m pip install " + " ".join(sorted(set(missing)))
        return render_template_string(MISSING_TEMPLATE, missing=missing, install_cmd=install_cmd, err=_import_error or "")
else:
    # All imports succeeded: define working app routes and model-loading behavior
    BUNDLE_PATH = os.path.join(APP_DIR, "internship_company_recommender.joblib")
    CSV_INPUT = os.path.join(APP_DIR, "internshipdata.csv")

    # create small templates in-memory (no separate files required)
    INDEX_HTML = """
    <!doctype html>
    <html>
      <head><meta charset="utf-8"><title>Intern Recommender</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      </head>
      <body class="bg-light">
      <div class="container py-5">
        <h1>Internship Recommendation System</h1>
        <form method="post" action="/predict" class="card p-4">
          <div class="row">
            <div class="col-md-4"><label>Student name</label><input name="Student name" class="form-control"></div>
            <div class="col-md-2"><label>Age</label><input name="Age" class="form-control" value="21"></div>
            <div class="col-md-3"><label>Gender</label><select name="Gender" class="form-select"><option>Male</option><option>Female</option><option>Other</option></select></div>
            <div class="col-md-3"><label>Location</label><input name="Location" class="form-control" value="Delhi"></div>
          </div>
          <div class="row mt-3">
            <div class="col-md-3"><label>CGPA</label><input name="CGPA" class="form-control" value="8.0"></div>
            <div class="col-md-3"><label>Work experience (months)</label><input name="Work experience" class="form-control" value="0"></div>
            <div class="col-md-3"><label>Company location</label><input name="Company location" class="form-control" value="Bangalore"></div>
            <div class="col-md-3"><label>Internship status</label><select name="Internship status" class="form-select"><option>NO</option><option>YES</option></select></div>
          </div>
          <div class="mt-3"><label>Technical skills (comma separated)</label><input name="Technical skills" class="form-control" placeholder="Python, ML"></div>
          <div class="mt-3"><label>Rating by company</label><input name="Rating by company" class="form-control" value="0"></div>
          <div class="mt-3"><button class="btn btn-primary">Recommend</button></div>
        </form>
      </div>
      </body>
    </html>
    """

    RESULT_HTML = """
    <!doctype html>
    <html>
      <head><meta charset="utf-8"><title>Result</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      </head><body class="bg-light">
      <div class="container py-5"><h2>Recommendation</h2>
      <div class="card p-4"><h3>Recommended Company: <span class="text-primary">{{company}}</span></h3>
      <p>{{notes}}</p>
      <a href="/" class="btn btn-secondary mt-3">Back</a></div></div></body></html>
    """

    # Model bundle variable and loader
    bundle = None

    def parse_experience_input(x):
        try:
            if isinstance(x, str) and x.strip().isdigit():
                return int(x.strip())
            m = re.search(r'(\d+)', str(x))
            if m:
                return int(m.group(1))
        except:
            pass
        return 0

    def safe_label_encode(le, value):
        try:
            if value in list(le.classes_):
                return int(le.transform([value])[0])
        except:
            pass
        return 0

    # Use before_request (Flask 3+) but ensure we only load once
    @app.before_request
    def ensure_bundle_loaded():
        global bundle
        if bundle is not None:
            return
        # Try to load pre-saved bundle
        if os.path.exists(BUNDLE_PATH):
            bundle_local = joblib.load(BUNDLE_PATH)
            bundle = bundle_local
            app.logger.info("Loaded model bundle from %s", BUNDLE_PATH)
            return
        # fallback: if CSV exists, train quick model
        if os.path.exists(CSV_INPUT):
            try:
                df = pd.read_csv(CSV_INPUT)
                df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median())
                df['CGPA'] = pd.to_numeric(df['CGPA'], errors='coerce').fillna(df['CGPA'].median())
                df['Rating by company'] = pd.to_numeric(df['Rating by company'], errors='coerce').fillna(0)
                df['Work experience months'] = df['Work experience'].apply(parse_experience_input)
                df['Internship status bin'] = df['Internship status'].astype(str).map({'YES':1,'NO':0}).fillna(0).astype(int)
                le_gender = LabelEncoder().fit(df['Gender'].astype(str)); df['Gender_enc'] = le_gender.transform(df['Gender'].astype(str))
                le_location = LabelEncoder().fit(df['Location'].astype(str)); df['Location_enc'] = le_location.transform(df['Location'].astype(str))
                le_company_loc = LabelEncoder().fit(df['Company location'].astype(str)); df['Company_location_enc'] = le_company_loc.transform(df['Company location'].astype(str))
                le_company = LabelEncoder().fit(df['Company name'].astype(str)); df['Company_enc'] = le_company.transform(df['Company name'].astype(str))
                vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
                skill_matrix = vectorizer.fit_transform(df['Technical skills'].astype(str))
                X_num = df[['Age','Gender_enc','CGPA','Location_enc','Company_location_enc','Work experience months','Internship status bin','Rating by company']].reset_index(drop=True)
                X_skills = pd.DataFrame(skill_matrix.toarray(), columns=[f'skill_{s}' for s in vectorizer.get_feature_names_out()])
                X = pd.concat([X_num, X_skills], axis=1); X.columns = X.columns.astype(str)
                y = df['Company_enc']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=200, random_state=42); model.fit(X_train, y_train)
                bundle_local = {'model': model, 'vectorizer': vectorizer, 'le_gender': le_gender, 'le_location': le_location, 'le_company_loc': le_company_loc, 'le_company': le_company, 'feature_columns': X.columns.tolist()}
                joblib.dump(bundle_local, BUNDLE_PATH)
                bundle = bundle_local
                app.logger.info("Trained and saved quick bundle to %s", BUNDLE_PATH)
                return
            except Exception as e:
                app.logger.error("Failed to train from CSV: %s", e)
                return

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML)

    @app.route("/predict", methods=["POST"])
    def predict():
        form = request.form
        student = {}
        student['Age'] = int(form.get('Age') or 21)
        student['Gender'] = form.get('Gender') or 'Male'
        student['Location'] = form.get('Location') or 'Delhi'
        student['CGPA'] = float(form.get('CGPA') or 8.0)
        student['Technical skills'] = form.get('Technical skills') or ''
        student['Work experience months'] = parse_experience_input(form.get('Work experience') or '0')
        student['Company location'] = form.get('Company location') or 'Bangalore'
        student['Rating by company'] = int(form.get('Rating by company') or 0)
        student['Internship status'] = form.get('Internship status') or 'NO'

        # prepare vector
        vec = bundle['vectorizer'].transform([student['Technical skills']])
        skill_cols = [f"skill_{s}" for s in bundle['vectorizer'].get_feature_names_out()]
        skills_df = pd.DataFrame(vec.toarray(), columns=skill_cols)
        le_gender = bundle['le_gender']; le_location = bundle['le_location']; le_company_loc = bundle['le_company_loc']; le_company = bundle['le_company']

        row = {}
        row['Age'] = student['Age']
        row['Gender_enc'] = safe_label_encode(le_gender, student['Gender'])
        row['CGPA'] = student['CGPA']
        row['Location_enc'] = safe_label_encode(le_location, student['Location'])
        row['Company_location_enc'] = safe_label_encode(le_company_loc, student['Company location'])
        row['Work experience months'] = student['Work experience months']
        row['Internship status bin'] = 1 if str(student['Internship status']).upper() == 'YES' else 0
        row['Rating by company'] = student['Rating by company']

        X_new = pd.DataFrame([row])
        X_new = pd.concat([X_new.reset_index(drop=True), skills_df], axis=1)
        for c in bundle['feature_columns']:
            if c not in X_new.columns:
                X_new[c] = 0
        X_new = X_new[bundle['feature_columns']]
        X_new.columns = X_new.columns.astype(str)

        pred_enc = bundle['model'].predict(X_new)[0]
        pred_company = le_company.inverse_transform([pred_enc])[0]
        notes = f"Predicted using Random Forest. Input mapped to {len(X_new.columns)} features."
        return render_template_string(RESULT_HTML, company=pred_company, notes=notes)

if __name__ == "__main__":
    # run app
    app.run(host="0.0.0.0", port=7860, debug=True)
