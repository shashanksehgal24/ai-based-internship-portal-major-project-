# train_intern_recommender_clean.py
import os
import re
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.simplefilter(action="ignore", category=FutureWarning)

########## CONFIG ##########
INPUT_PATH_CSV = "internshipdata.csv"   # your CSV filename
INPUT_PATH_XLSX = "internship_data.xlsx"
MODEL_OUT_PATH = "internship_company_recommender.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
############################

def read_input(path_csv=INPUT_PATH_CSV, path_xlsx=INPUT_PATH_XLSX):
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
        print(f"Loaded CSV: {path_csv}")
    elif os.path.exists(path_xlsx):
        df = pd.read_excel(path_xlsx)
        print(f"Loaded Excel: {path_xlsx}")
    else:
        raise FileNotFoundError(f"Neither {path_csv} nor {path_xlsx} were found.")
    return df

def parse_experience(x):
    if pd.isna(x):
        return 0
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        m = re.search(r'(\d+)', x)
        if m:
            return int(m.group(1))
    return 0

def safe_label_encode(le: LabelEncoder, value, default_index=0):
    try:
        if value in le.classes_:
            return int(le.transform([value])[0])
    except Exception:
        pass
    return int(default_index)

def prepare_features(df):
    required = ["Student name","Age","Gender","Location","CGPA","Technical skills",
                "Work experience","Company name","Company location",
                "Skills required by company","Internship status","Rating by company"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df.copy()  # ensure we work on a copy

    # numeric coercion (no inplace)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["CGPA"] = pd.to_numeric(df["CGPA"], errors="coerce")
    df["Rating by company"] = pd.to_numeric(df["Rating by company"], errors="coerce")

    # fill numeric NaNs by assignment (avoid chained-inplace warnings)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["CGPA"] = df["CGPA"].fillna(df["CGPA"].median())
    df["Rating by company"] = df["Rating by company"].fillna(0)

    # parse work experience into months integer
    df["Work experience months"] = df["Work experience"].apply(parse_experience)

    # internship status binary
    df["Internship status bin"] = df["Internship status"].astype(str).map({"YES": 1, "NO": 0})
    df["Internship status bin"] = df["Internship status bin"].fillna(0).astype(int)

    # label encoders
    le_gender = LabelEncoder().fit(df["Gender"].astype(str))
    df["Gender_enc"] = le_gender.transform(df["Gender"].astype(str))

    le_location = LabelEncoder().fit(df["Location"].astype(str))
    df["Location_enc"] = le_location.transform(df["Location"].astype(str))

    le_company_loc = LabelEncoder().fit(df["Company location"].astype(str))
    df["Company_location_enc"] = le_company_loc.transform(df["Company location"].astype(str))

    le_company = LabelEncoder().fit(df["Company name"].astype(str))
    df["Company_enc"] = le_company.transform(df["Company name"].astype(str))

    # vectorize skills
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    skill_matrix = vectorizer.fit_transform(df["Technical skills"].astype(str))

    numeric_cols = [
        "Age",
        "Gender_enc",
        "CGPA",
        "Location_enc",
        "Company_location_enc",
        "Work experience months",
        "Internship status bin",
        "Rating by company"
    ]

    X_numeric = df[numeric_cols].reset_index(drop=True)
    X_skills = pd.DataFrame(skill_matrix.toarray(),
                            columns=[f"skill_{s}" for s in vectorizer.get_feature_names_out()])

    X = pd.concat([X_numeric, X_skills], axis=1)

    # ensure all column names are strings (fix for sklearn)
    X.columns = X.columns.astype(str)

    y = df["Company_enc"]

    encoders = {
        "le_gender": le_gender,
        "le_location": le_location,
        "le_company_loc": le_company_loc,
        "le_company": le_company
    }

    return X, y, vectorizer, encoders, df

def train_and_save(X, y, vectorizer, encoders, model_out_path=MODEL_OUT_PATH):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "le_gender": encoders["le_gender"],
        "le_location": encoders["le_location"],
        "le_company_loc": encoders["le_company_loc"],
        "le_company": encoders["le_company"],
        "feature_columns": X.columns.tolist()
    }

    joblib.dump(bundle, model_out_path)
    print(f"Model bundle saved to: {model_out_path}")
    print(f"Test accuracy: {acc*100:.2f}%")
    return bundle, acc

def predict_company_for_student_bundle(bundle, student_dict):
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    le_gender = bundle["le_gender"]
    le_location = bundle["le_location"]
    le_company_loc = bundle["le_company_loc"]
    le_company = bundle["le_company"]
    feature_columns = bundle["feature_columns"]

    row = {}
    row["Age"] = student_dict.get("Age", 21)
    row["Gender_enc"] = safe_label_encode(le_gender, str(student_dict.get("Gender", "")), default_index=0)
    row["CGPA"] = student_dict.get("CGPA", 0)
    row["Location_enc"] = safe_label_encode(le_location, str(student_dict.get("Location", "")), default_index=0)
    row["Company_location_enc"] = safe_label_encode(le_company_loc, str(student_dict.get("Company location", "")), default_index=0)
    row["Work experience months"] = int(student_dict.get("Work experience months", 0))
    row["Internship status bin"] = 1 if str(student_dict.get("Internship status", "NO")).upper() == "YES" else 0
    row["Rating by company"] = int(student_dict.get("Rating by company", 0))

    skills_vec = vectorizer.transform([student_dict.get("Technical skills", "")])
    skills_df = pd.DataFrame(skills_vec.toarray(), columns=[f"skill_{s}" for s in vectorizer.get_feature_names_out()])

    X_new = pd.DataFrame([row])
    X_new = pd.concat([X_new.reset_index(drop=True), skills_df], axis=1)

    for col in feature_columns:
        if col not in X_new.columns:
            X_new[col] = 0

    X_new = X_new[feature_columns]
    X_new.columns = X_new.columns.astype(str)

    pred_enc = model.predict(X_new)[0]
    pred_company = le_company.inverse_transform([pred_enc])[0]
    return pred_company

if __name__ == "__main__":
    df = read_input()
    X, y, vectorizer, encoders, df_full = prepare_features(df)
    bundle, acc = train_and_save(X, y, vectorizer, encoders, MODEL_OUT_PATH)

    print("\nSample predictions on some dataset rows:")
    for i in range(min(3, len(df_full))):
        sample = {
            "Age": int(df_full.iloc[i]["Age"]),
            "Gender": df_full.iloc[i]["Gender"],
            "Location": df_full.iloc[i]["Location"],
            "CGPA": float(df_full.iloc[i]["CGPA"]),
            "Technical skills": df_full.iloc[i]["Technical skills"],
            "Work experience months": int(parse_experience(df_full.iloc[i]["Work experience"])),
            "Company location": df_full.iloc[i]["Company location"],
            "Rating by company": int(df_full.iloc[i]["Rating by company"]),
            "Internship status": df_full.iloc[i]["Internship status"]
        }
        predicted = predict_company_for_student_bundle(bundle, sample)
        print(f"Student {df_full.iloc[i]['Student name']} -> predicted company: {predicted}")

    print("\nDone. Use joblib.load to load", MODEL_OUT_PATH)
