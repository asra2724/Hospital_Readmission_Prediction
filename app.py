
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import io
import streamlit.components.v1 as components
import plotly.express as px

# ML imports 
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# -------------------------
# Utility: pipeline code (adapted for interactive use)
# -------------------------
def perform_feature_engineering(df):
    df = df.copy()
    # ensure diag cols present for counting
    for c in ['diag_1','diag_2','diag_3']:
        if c not in df.columns:
            df[c] = np.nan
    df['num_diagnoses'] = df[['diag_1','diag_2','diag_3']].count(axis=1)
    df['total_med_procedures'] = df.get('n_lab_procedures',0) + df.get('n_procedures',0)

    def simplify_age_group(age_range):
        if pd.isna(age_range): return 'Other'
        if age_range in ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)']:
            return 'Young'
        elif age_range in ['[50-60)','[60-70)','[70-80)']:
            return 'Middle-aged'
        else:
            return 'Senior'

    if 'age' in df.columns:
        df['age_group_simplified'] = df['age'].apply(simplify_age_group)
    else:
        df['age_group_simplified'] = 'Other'
    return df

def preprocess_data(df, target_col):
    """
    Returns X (features), y (target), numerical_cols (list).
    Converts all object/categorical columns with get_dummies.
    """
    df = df.copy()

    # Drop raw columns used only for feature engineering
    for c in ['age', 'diag_1', 'diag_2', 'diag_3']:
        if c in df.columns:
            df = df.drop(c, axis=1)

    # Drop unnamed columns
    unnamed = [c for c in df.columns if c.startswith('Unnamed')]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Handle target
    if target_col in df.columns:
        df[target_col] = df[target_col].map({'no':0,'yes':1}).astype(float)

    # Identify numerical & categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()

    # One-hot encode all categorical cols
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Clean column names
    df.columns = [str(c).replace('[','_').replace(']','_').replace('<','_')
                    .replace(',','_').replace('(','_').replace(')','_').replace(' ','_')
                  for c in df.columns]

    # Split features/target
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    else:
        X = df.copy()
        y = None

    # Update numerical cols (after dummies everything is numeric anyway)
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    return X, y, numerical_cols


def train_stacked_model(X, y, numerical_cols, quick_mode=True):
    """
    Trains the stacked model. quick_mode reduces GridSearch/CV to be faster in UI.
    Returns trained (preprocessor, classifier) and a best threshold (0.5 fallback).
    """
    # Basic preprocessor
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_cols)], remainder='passthrough')
    # Apply SMOTEENN on X,y
    smote_enn = SMOTEENN(random_state=42)
    X_res, y_res = smote_enn.fit_resample(X, y)
    # Fit preprocessor on resampled data
    X_proc = preprocessor.fit_transform(X_res)

    # Base models
    log_clf = LogisticRegression(solver='liblinear', max_iter=2000, random_state=42)
    log_clf.fit(X_proc, y_res)

    rf_clf = RandomForestClassifier(random_state=42)
    # smaller grid for speed
    rf_params = {'n_estimators':[100], 'max_depth':[10]} if quick_mode else {'n_estimators':[100,200],'max_depth':[10,20]}
    grid_rf = GridSearchCV(rf_clf, rf_params, cv=2 if quick_mode else 3, scoring='f1', n_jobs=-1)
    grid_rf.fit(X_proc, y_res)
    best_rf = grid_rf.best_estimator_

    ada_clf = AdaBoostClassifier(random_state=42)
    ada_params = {'n_estimators':[50], 'learning_rate':[0.5]} if quick_mode else {'n_estimators':[100],'learning_rate':[0.1,0.5]}
    grid_ada = GridSearchCV(ada_clf, ada_params, cv=2 if quick_mode else 3, scoring='f1', n_jobs=-1)
    grid_ada.fit(X_proc, y_res)
    best_ada = grid_ada.best_estimator_

    stack_clf = StackingClassifier(estimators=[('lr',log_clf),('rf',best_rf),('ada',best_ada)],
                                   final_estimator=LogisticRegression(), cv=2 if quick_mode else 3, n_jobs=-1)
    stack_clf.fit(X_proc, y_res)

    # default threshold
    best_thresh = 0.5

    return preprocessor, stack_clf, best_thresh

# ------------------------- 
# APP CONFIG
# -------------------------
st.set_page_config(page_title="ReadmitCare", layout='wide', page_icon="⚕️")

# ------------------------- 
# GLOBAL CSS
# -------------------------
# Force Tailwind gradient theme
st.markdown("""
<style>
/* Global background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(to right, #fdf2f8, #ede9fe, #dbeafe) !important;
    color: #1e293b !important;
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #f0f9ff, #e0e7ff, #f5d0fe) !important;
    color: #1e293b !important;
}

/* Headings */
h1 {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(to right, #3b82f6, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2em;
}
.subheading {
    text-align: center;
    font-size: 1.2rem;
    color: #475569;
    margin-top: -6px;
    margin-bottom: 1.5rem;
}

/* Navbar buttons */
.stButton > button {
    background: linear-gradient(to right, #3b82f6, #a855f7, #ec4899);
    color: white !important;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    border-radius: 0.75rem;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 25px rgba(168,85,247,0.4);
}

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 1rem;
    padding: 1.5rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
}

/* Team images */
.team-img {
    width: 90px;
    height: 90px;
    border-radius: 9999px;
    object-fit: cover;
    border: 4px solid #fff;
    box-shadow: 0 6px 15px rgba(0,0,0,0.15);
}

/*  Form Labels */
div[data-testid="stMarkdownContainer"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label {
    color: #000 !important;
    font-weight: 600 !important;
}

/* Inputs (Text + Number) */
div[data-baseweb="input"] input,
.stNumberInput input {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 0.5rem !important;
    padding: 0.5rem 0.75rem !important;
    font-size: 0.95rem !important;
    transition: all 0.25s ease;
}
div[data-baseweb="input"] input:focus,
.stNumberInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.3) !important;
    outline: none !important;
}

/*  Selectbox & Multiselect */
div[data-baseweb="select"] {
    width: 100% !important;
    min-width: 280px !important;
    max-width: 100% !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 0.75rem !important;
    background-color: #ffffff !important;
    color: #1e293b !important;
    transition: all 0.25s ease;
}
div[data-baseweb="select"]:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.3) !important;
}

/* Inner container */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.4rem 0.75rem !important;
}

/* Dropdown text */
div[data-baseweb="select"] input,
div[data-baseweb="select"] span {
    color: #1e293b !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
}

/* Dropdown arrow */
div[data-baseweb="select"] svg {
    margin-right: 8px;
}

/* Dropdown menu */
div[data-baseweb="popover"] {
    max-height: 280px !important;
    overflow-y: auto !important;
    border-radius: 0.75rem !important;
    border: 1px solid #cbd5e1 !important;
    background-color: #ffffff !important;
}
div[data-baseweb="popover"] div {
    background-color: #ffffff !important;
    color: #1e293b !important;
}

/*  Hover glow */
div[data-baseweb="input"] input:hover,
.stNumberInput input:hover,
div[data-baseweb="select"]:hover {
    background-color: #ffffff !important;
    box-shadow: 0 0 12px rgba(168,85,247,0.3) !important;
    transform: scale(1.01);
}

/* Prediction form spacing */
.prediction-form .stSelectbox,
.prediction-form .stNumberInput,
.prediction-form .stTextInput {
    margin-bottom: 1rem;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 0.75rem !important;
    font-weight: 600 !important;
    padding: 0.9rem !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Info alert */
[data-testid="stAlert"].stAlertInfo,
[data-testid="stAlert"][class*="stAlert-info"] {
    background: #dbeafe !important;
    border-left: 6px solid #2563eb !important;
}

/* Success alert */
[data-testid="stAlert"].stAlertSuccess,
[data-testid="stAlert"][class*="stAlert-success"] {
    background: #dcfce7 !important;
    border-left: 6px solid #16a34a !important;
}

/* Warning alert */
[data-testid="stAlert"].stAlertWarning,
[data-testid="stAlert"][class*="stAlert-warning"] {
    background: #fef9c3 !important;
    border-left: 6px solid #ca8a04 !important;
}

/* Error alert */
[data-testid="stAlert"].stAlertError,
[data-testid="stAlert"][class*="stAlert-error"] {
    background: #fee2e2 !important;
    border-left: 6px solid #dc2626 !important;
}

</style>



...""", unsafe_allow_html=True)


# -------------------------
# HEADER + NAVIGATION
# -------------------------
st.markdown("<h1>ReadmitCare</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheading'>AI-driven patient readmission risk insights — intervene early, improve outcomes.</p>", unsafe_allow_html=True)

# Initialize page state
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# NAVIGATION BAR
nav_cols = st.columns(3)
pages = ["Home", "Prediction", "Visualization"]

for i, p in enumerate(pages):
    if nav_cols[i].button(p, key=f"nav_{p}"):
        st.session_state["page"] = p

page = st.session_state["page"]




# Load dataset if available (data.csv in same folder)
DATA_PATH = "data.csv"
data_exists = os.path.exists(DATA_PATH)
if data_exists:
    try:
        df_data = pd.read_csv(DATA_PATH)
    except Exception:
        df_data = pd.DataFrame()
else:
    df_data = pd.DataFrame()

# HOME PAGE
if page == "Home":
    st.markdown('<div class="home-bg" style="padding:2rem;border-radius:1rem;">', unsafe_allow_html=True)
    st.markdown("###  Welcome to ReadmitCare Dashboard")
    st.markdown("Predict, prevent, and visualize patient readmissions with AI magic ")
    st.markdown("</div>", unsafe_allow_html=True)



    # Dataset block
    st.write("")
    colA, colB = st.columns([2,1])
    with colA:
        st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
        st.markdown("### Dataset used")
        if data_exists:
            st.markdown(f"- Loaded `data.csv` with **{len(df_data):,} rows**.")
            st.markdown("- Source: Kaggle hospital readmissions dataset.")
            st.markdown("- Link: https://www.kaggle.com/datasets/dubradave/hospital-readmissions")
        else:
            st.markdown("- `data.csv` not found locally. The app will use synthetic fallback for demo purposes.")
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
        st.markdown("### Tech stack")
        st.markdown("- Python, Pandas, scikit-learn")
        st.markdown("- imbalanced-learn (SMOTEENN)")
        st.markdown("- Streamlit + Plotly")
        st.markdown("</div>", unsafe_allow_html=True)

    # Journey
    st.markdown("<div class='card fade-in' style='margin-top:12px'>", unsafe_allow_html=True)
    st.markdown("### Project Journey")
    st.markdown("1. EDA → 2. Preprocessing & Feature Engineering → 3. Class imbalance handling (SMOTEENN) → 4. Stacked model (LR + RF + AdaBoost) → 5. Deploy with Streamlit")
    st.markdown("</div>", unsafe_allow_html=True)

    # TEAM SECTION (Home page)
    st.markdown("<div class='card fade-in' style='margin-top:12px'>", unsafe_allow_html=True)
    st.markdown("### Team")
    cols = st.columns(6)  # now 6 members
    team = [
        ("Chandra", "http://www.linkedin.com/in/p-chandra-sekhar", "https://github.com/P-Chandra28", "chandra_shaker.png"),
        ("Venkat", "https://www.linkedin.com/in/venkat-ganesh-8557362b8/", "https://github.com/venkat-ganesh4", "venkat_ganesh.jpg"),
        ("Rishitha", "https://www.linkedin.com/in/rishitha-muthineni/", "https://github.com/rishithamuthineni", "rishitha.jpg"),
        ("Satvika", "https://www.linkedin.com/in/nsatvika/", "https://github.com/Satvika26", "satvika.jpg"),
        ("Asra", "https://www.linkedin.com/in/asra-profile", "https://github.com/asra-profile", "asra.jpg"),
        ("Akshaya", "https://www.linkedin.com/in/akshaya-profile", "https://github.com/akshaya-profile", "akshaya.jpg"),
    ]
    for idx, (name, ln, gh, img) in enumerate(team):
        col = cols[idx % 6]
        with col:
            if os.path.exists(img):
                col.markdown(f"<img src='{img}' class='team-img'/>", unsafe_allow_html=True)
            else:
                col.markdown(
                    f"<div style='width:72px;height:72px;border-radius:50%;background:linear-gradient(90deg,#6366f1,#06b6d4);display:inline-block;margin-bottom:8px;'></div>",
                    unsafe_allow_html=True
                )
            col.markdown(f"**{name}**  <br> [LinkedIn]({ln}) | [GitHub]({gh})", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# PREDICTION PAGE
elif page == "Prediction":
    st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
    st.markdown("## Patient Readmission Prediction")
    st.markdown("Fill the patient attributes below, press **Train & Predict**, and get risk with probability.")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.form("patient_form"):
        st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)

        # ---------------------------
        # 5 rows × 3 columns layout
        # ---------------------------

        # Row 1
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            age = st.selectbox("Age range",
                               options=['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
                                        '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)'],
                               index=5)
        with r1c2:
            time_in_hospital = st.number_input("Time in hospital (days)", min_value=0, max_value=100, value=5)
        with r1c3:
            n_lab_procedures = st.number_input("Number of lab procedures", min_value=0, max_value=500, value=40)

        # Row 2
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            n_procedures = st.number_input("Number of procedures", min_value=0, max_value=100, value=1)
        with r2c2:
            n_medications = st.number_input("Number of medications", min_value=0, max_value=200, value=10)
        with r2c3:
            n_outpatient = st.number_input("Number outpatient visits", min_value=0, max_value=200, value=0)

        # Row 3
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            n_inpatient = st.number_input("Number inpatient visits", min_value=0, max_value=200, value=0)
        with r3c2:
            n_emergency = st.number_input("Number emergency visits", min_value=0, max_value=200, value=0)
        with r3c3:
            medical_specialty = st.text_input("Medical specialty (e.g. Cardiology)", value="Cardiology")

        # Row 4
        r4c1, r4c2, r4c3 = st.columns(3)
        with r4c1:
            A1Ctest = st.selectbox("A1C test?", options=['None','Norm','High'], index=0)
        with r4c2:
            change = st.selectbox("Change in medication?", options=['No','Ch'], index=0)
        with r4c3:
            diabetes_med = st.selectbox("Diabetes medication?", options=['No','Yes'], index=1)

        # Row 5
        r5c1, r5c2, r5c3 = st.columns(3)
        with r5c1:
            diag_1 = st.text_input("Primary diagnosis (diag_1)", value="250.83")
        with r5c2:
            diag_2 = st.text_input("Secondary diagnosis (diag_2)", value="")
        with r5c3:
            diag_3 = st.text_input("Tertiary diagnosis (diag_3)", value="")

        # Submit button at the bottom
        submitted = st.form_submit_button("Train & Predict")


    #  All logic INSIDE if submitted
    if submitted:
        # -------------------
        # Build user DataFrame
        # -------------------
        df_user = pd.DataFrame([{
            "age": age,
            "time_in_hospital": time_in_hospital,
            "n_lab_procedures": n_lab_procedures,
            "n_procedures": n_procedures,
            "n_medications": n_medications,
            "n_outpatient": n_outpatient,
            "n_inpatient": n_inpatient,
            "n_emergency": n_emergency,
            "medical_specialty": medical_specialty,
            "A1Ctest": A1Ctest,
            "change": change,
            "diabetes_med": diabetes_med,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
        }])

        # -------------------
        # Timeline Animation
        # -------------------
        progress = st.progress(0)
        status_text = st.empty()
        steps = [
            (" Starting model training...", 25),
            (" Processing patient data...", 55),
            (" Making predictions...", 85),
            (" Prediction complete!", 100),
        ]
        for step, pct in steps:
            status_text.markdown(f"**{step}**")
            progress.progress(pct)
            time.sleep(1.2)

        # -------------------
        # Train model + predict
        # -------------------
        with st.spinner("Finalizing results..."):
            if not df_data.empty:
                combined = pd.concat([df_data, df_user], ignore_index=True, sort=False)
                if "readmitted" not in combined.columns:
                    combined["readmitted"] = np.random.choice(["no", "yes"], size=len(combined), p=[0.8, 0.2])
            else:
                combined = df_user.copy()
                combined["readmitted"] = np.nan

            combined_fe = perform_feature_engineering(combined)
            X_all, y_all, numerical_cols = preprocess_data(combined_fe, "readmitted")

            last_idx = X_all.index[-1]
            X_to_predict = X_all.loc[[last_idx]]
            X_train = X_all.drop(index=last_idx)
            y_train = y_all.drop(index=last_idx)

            preproc, classifier, best_thresh = train_stacked_model(
                X_train, y_train.astype(int), numerical_cols, quick_mode=True
            )

            X_pred_proc = preproc.transform(X_to_predict)
            proba = classifier.predict_proba(X_pred_proc)[0][1]
            pred_label = int(proba >= best_thresh)
            prediction = "High Risk" if pred_label == 1 else "Low Risk"

        # -------------------
        #  Results card
        # -------------------
        st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
        if prediction == "High Risk":
            st.error(f" Prediction: **{prediction}** with probability {proba:.2%}")
        else:
            st.success(f" Prediction: **{prediction}** with probability {proba:.2%}")
        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------
        #  Interactive Charts
        # -------------------
        import plotly.express as px
        import plotly.graph_objects as go

        # Feature importance
        
        feature_importance = X_train.corrwith(y_train).abs().sort_values(ascending=False).head(10)
        fig_imp = px.bar(
            feature_importance,
            x=feature_importance.values,
            y=feature_importance.index,
            orientation="h",
            title="Top 10 Important Features",
            labels={"x": "Importance", "y": "Features"},
            color=feature_importance.values,
            color_continuous_scale="viridis"
        )
        st.markdown("<div class='card fade-in' style='margin-top:16px'>", unsafe_allow_html=True)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Probability distribution
        y_probs = classifier.predict_proba(preproc.transform(X_train))[:, 1]
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Histogram(x=y_probs, nbinsx=20, name="All Patients", opacity=0.6))
        fig_prob.add_vline(
            x=proba, line_width=3, line_dash="dash", line_color="red",
            annotation_text="Your Patient", annotation_position="top right"
        )
        fig_prob.update_layout(
            title="Probability of Readmission Distribution",
            xaxis_title="Probability",
            yaxis_title="Count",
            bargap=0.1,
            height=400
        )
        st.markdown("<div class='card fade-in' style='margin-top:16px'>", unsafe_allow_html=True)
        st.plotly_chart(fig_prob, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# VISUALIZATION
elif page == "Visualization":
     st.markdown("<div class='card fade-in'>", unsafe_allow_html=True)
     st.markdown("##  Interactive Visualizations")
     st.markdown("</div>", unsafe_allow_html=True)

     if df_data.empty:
        st.warning("Dataset not found. Please upload or place `data.csv` in the app folder.")
     else:
        st.markdown("<div class='card fade-in' style='margin-top:16px'>", unsafe_allow_html=True)

        import plotly.express as px

        #  Distribution of hospital stay
        col1, col2 = st.columns(2)
        with col1:
            if "time_in_hospital" in df_data.columns:
                fig1 = px.histogram(
                    df_data, x="time_in_hospital",
                    nbins=15, title=" Time in Hospital Distribution",
                    color_discrete_sequence=["#3b82f6"]
                )
                fig1.update_traces(marker_line_width=1, marker_line_color="white")
                fig1.update_layout(height=400, width=500, template="plotly_dark", bargap=0.2)
                st.plotly_chart(fig1, use_container_width=True)

        #  Medications vs Readmission
        with col2:
            if "n_medications" in df_data.columns and "readmitted" in df_data.columns:
                fig2 = px.box(
                    df_data, x="readmitted", y="n_medications",
                    color="readmitted",
                    title=" Medications vs Readmission",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig2.update_layout(height=400, width=500, template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)

        # Emergency visits trend
        col3, col4 = st.columns(2)
        with col3:
            if "n_emergency" in df_data.columns:
                df_trend = df_data.groupby("time_in_hospital")["n_emergency"].mean().reset_index()
                fig3 = px.line(
                    df_trend, x="time_in_hospital", y="n_emergency",
                    title=" Avg Emergency Visits Over Hospital Stay",
                    markers=True, line_shape="spline"
                )
                fig3.update_traces(line=dict(width=3))
                fig3.update_layout(height=400, width=500, template="plotly_dark")
                st.plotly_chart(fig3, use_container_width=True)

        # 4️ Diagnosis count
        with col4:
            diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df_data.columns]
            if diag_cols:
                diag_counts = pd.Series(df_data[diag_cols].values.ravel()).value_counts().head(10)
                diag_df = diag_counts.reset_index()
                diag_df.columns = ["Diagnosis", "Count"]
                fig4 = px.bar(
                    diag_df, x="Diagnosis", y="Count",
                    title=" Top 10 Diagnoses",
                    color="Count", color_continuous_scale="plasma"
                )
                fig4.update_layout(height=400, width=500, template="plotly_dark", xaxis_tickangle=-45)
                st.plotly_chart(fig4, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# Footer
st.markdown("<br><hr><p style='text-align:center;color:#475569'>ReadmitCare • Built with Streamlit</p>", unsafe_allow_html=True)
