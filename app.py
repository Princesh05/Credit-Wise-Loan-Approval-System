import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LoanWise AI | Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# PROFESSIONAL CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* ---------- Global Styling ---------- */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 30%),
        linear-gradient(135deg, #f8fafc 0%, #eef4ff 100%);
    color: #0f172a;
}

.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ---------- Header ---------- */

.hero {
    position: relative;
    overflow: hidden;
    padding: 42px 45px;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 64, 175, 0.96)),
        radial-gradient(circle at top right, #38bdf8, transparent 40%);
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.20);
    color: white;
    margin-bottom: 28px;
}

.hero::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -80px;
    top: -100px;
    border-radius: 50%;
    background: rgba(56, 189, 248, 0.18);
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 30px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: #bae6fd;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 15px;
}

.hero h1 {
    font-size: clamp(32px, 5vw, 55px);
    line-height: 1.1;
    margin: 0 0 12px 0;
    font-weight: 800;
    letter-spacing: -1.5px;
}

.hero p {
    max-width: 720px;
    color: #dbeafe;
    font-size: 17px;
    line-height: 1.7;
    margin: 0;
}

/* ---------- Cards ---------- */

.info-card,
.form-card,
.result-card,
.metric-card,
.tip-card {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
    backdrop-filter: blur(8px);
}

.info-card {
    margin-bottom: 25px;
    border-left: 5px solid #2563eb;
}

.info-card h3 {
    margin-top: 0;
    color: #0f172a;
    font-size: 21px;
}

.info-card p {
    color: #475569;
    line-height: 1.7;
    margin-bottom: 0;
}

.form-card {
    margin-top: 10px;
}

.section-title {
    color: #0f172a;
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 4px;
}

.section-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 23px;
}

/* ---------- Streamlit Inputs ---------- */

label {
    color: #334155 !important;
    font-weight: 600 !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
    background-color: #ffffff !important;
    min-height: 45px;
    transition: all 0.2s ease-in-out;
}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.14) !important;
}

/* ---------- Button ---------- */

.stButton > button {
    width: 100%;
    min-height: 54px;
    border: none;
    border-radius: 14px;
    color: white;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 0.2px;
    box-shadow: 0 10px 22px rgba(37, 99, 235, 0.25);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    color: white;
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    transform: translateY(-2px);
    box-shadow: 0 14px 28px rgba(37, 99, 235, 0.35);
}

.stButton > button:active {
    transform: translateY(0);
}

/* ---------- Results ---------- */

.approved-result,
.rejected-result {
    padding: 34px 25px;
    border-radius: 22px;
    text-align: center;
    margin: 18px 0 25px 0;
}

.approved-result {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #22c55e;
    color: #166534;
}

.rejected-result {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 2px solid #f43f5e;
    color: #9f1239;
}

.approved-result h1,
.rejected-result h1 {
    font-size: 32px;
    margin: 0 0 10px 0;
}

.approved-result p,
.rejected-result p {
    margin: 0;
    font-size: 15px;
    line-height: 1.6;
}

/* ---------- Confidence ---------- */

.confidence-card {
    padding: 22px;
    border-radius: 18px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin-bottom: 22px;
}

.confidence-label {
    display: flex;
    justify-content: space-between;
    color: #334155;
    font-weight: 700;
    margin-bottom: 10px;
}

.progress-background {
    width: 100%;
    height: 13px;
    border-radius: 20px;
    background: #e2e8f0;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
}

/* ---------- Metrics ---------- */

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 5px 15px rgba(15, 23, 42, 0.05);
}

div[data-testid="stMetricLabel"] {
    color: #64748b !important;
}

div[data-testid="stMetricValue"] {
    color: #0f172a !important;
}

/* ---------- Tips ---------- */

.tip-card {
    background: linear-gradient(135deg, #eff6ff, #f0f9ff);
    border-color: #bfdbfe;
    margin-top: 24px;
}

.tip-card h4 {
    color: #1e3a8a;
    margin-top: 0;
}

.tip-card p {
    color: #475569;
    line-height: 1.6;
}

/* ---------- Footer ---------- */

.footer {
    text-align: center;
    color: #64748b;
    padding: 28px 10px 10px 10px;
    font-size: 13px;
}

.footer strong {
    color: #1e40af;
}

/* ---------- Mobile Responsive ---------- */

@media only screen and (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .hero {
        padding: 30px 24px;
        border-radius: 22px;
    }

    .hero h1 {
        font-size: 35px;
    }

    .hero p {
        font-size: 15px;
    }

    .info-card,
    .form-card,
    .result-card {
        padding: 19px;
        border-radius: 18px;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    artifact = joblib.load("loan_approval_model.pkl")

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]

    return model, scaler, feature_names


try:
    model, scaler, feature_names = load_model()

except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.info(
        "Make sure that loan_approval_model.pkl is present in the same "
        "folder as this Streamlit application."
    )
    st.stop()

# =========================================================
# MODEL VALIDATION
# =========================================================

if isinstance(model, type):
    st.error(
        "The saved file contains a model class instead of a trained model object."
    )
    st.info(
        "Train your GaussianNB model first, then save the trained object using joblib."
    )
    st.stop()

# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-badge">🤖 MACHINE LEARNING POWERED</div>
        <h1>CreditWise Loan Approval System</h1>
        <p>
            A smart and simple loan approval prediction system that analyzes
            applicant information and provides a machine-learning-based decision.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INTRODUCTION
# =========================================================

st.markdown("""
<div class="info-card">
    <h3>💡 Make a Smarter Financial Decision</h3>
    <p>
        Enter the applicant's personal, employment, income, loan, and credit
        information. The trained machine learning model will estimate whether
        the application is likely to be approved.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# MAIN FORM
# =========================================================

st.markdown(
    '<div class="section-title">👤 Applicant Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Provide accurate details for a better prediction.</div>',
    unsafe_allow_html=True
)

with st.form("loan_prediction_form"):

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🧾 Personal Information")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
            help="Applicant's age must be between 18 and 100 years."
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Divorced", "Married", "Single"]
        )

        num_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

    with col2:
        st.markdown("#### 💰 Financial Information")

        annual_income = st.number_input(
            "Annual Income (₹)",
            min_value=0,
            value=50000,
            step=1000,
            format="%d"
        )

        loan_amount = st.number_input(
            "Requested Loan Amount (₹)",
            min_value=0,
            value=20000,
            step=1000,
            format="%d"
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650,
            step=1,
            help="Typical credit scores range from 300 to 900."
        )

        existing_loans_count = st.number_input(
            "Number of Existing Loans",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

    st.markdown("#### 💼 Employment Information")

    employment_status = st.selectbox(
        "Employment Status",
        ["Employed", "Self-employed", "Unemployed"]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.form_submit_button(
        "🔍 Analyze Loan Application",
        use_container_width=True
    )

# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Basic validation
    if annual_income <= 0:
        st.warning("Please enter a valid annual income.")
        st.stop()

    if loan_amount <= 0:
        st.warning("Please enter a valid loan amount.")
        st.stop()

    # Categorical encoding
    gender_1 = 1 if gender == "Male" else 0

    marital_status_1 = 1 if marital_status == "Married" else 0
    marital_status_2 = 1 if marital_status == "Single" else 0

    employment_status_1 = 1 if employment_status == "Self-employed" else 0
    employment_status_2 = 1 if employment_status == "Unemployed" else 0

    # Feature engineering
    num_dependents_sq = num_dependents ** 2
    credit_score_sq = credit_score ** 2

    # Create input data
    input_data = pd.DataFrame([{
        "age": age,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "existing_loans_count": existing_loans_count,
        "gender_1": gender_1,
        "marital_status_1": marital_status_1,
        "marital_status_2": marital_status_2,
        "employment_status_1": employment_status_1,
        "employment_status_2": employment_status_2,
        "num_dependents_sq": num_dependents_sq,
        "credit_score_sq": credit_score_sq
    }])

    # Check required features
    missing_features = [
        feature for feature in feature_names
        if feature not in input_data.columns
    ]

    if missing_features:
        st.error("The following features are missing from the application input:")
        st.code(", ".join(missing_features))
        st.stop()

    # Put features in exact training order
    input_data = input_data[feature_names]

    try:
        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        # Prediction probability
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_scaled)[0]

            if hasattr(model, "classes_"):
                prediction_index = list(model.classes_).index(prediction)
                probability = float(probabilities[prediction_index] * 100)

    except Exception as e:
        st.error("An error occurred while generating the prediction.")
        st.code(str(e))
        st.stop()

    # =====================================================
    # RESULT SECTION
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">📋 Prediction Result</div>',
        unsafe_allow_html=True
    )

    if int(prediction) == 1:

        st.markdown("""
        <div class="approved-result">
            <h1>✅ Loan Approved</h1>
            <p>
                The model predicts that this application is likely to be approved
                based on the provided information.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

    else:

        st.markdown("""
        <div class="rejected-result">
            <h1>❌ Loan Not Approved</h1>
            <p>
                The model predicts that this application may not be approved
                based on the provided information.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # CONFIDENCE DISPLAY
    # =====================================================

    if probability is not None:

        confidence = max(0, min(100, probability))

        st.markdown(f"""
        <div class="confidence-card">
            <div class="confidence-label">
                <span>🎯 Model Confidence</span>
                <span>{confidence:.2f}%</span>
            </div>
            <div class="progress-background">
                <div class="progress-fill" style="width: {confidence}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # SUMMARY CARDS
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Application Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">A quick overview of the submitted applicant details.</div>',
        unsafe_allow_html=True
    )

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.metric("Applicant Age", f"{age} years")
        st.metric("Gender", gender)

    with summary_col2:
        st.metric("Annual Income", f"₹{annual_income:,.0f}")
        st.metric("Employment", employment_status)

    with summary_col3:
        st.metric("Loan Amount", f"₹{loan_amount:,.0f}")
        st.metric("Credit Score", credit_score)

    with summary_col4:
        st.metric("Dependents", num_dependents)
        st.metric("Existing Loans", existing_loans_count)

    # =====================================================
    # MODEL INSIGHT
    # =====================================================

    st.markdown("""
    <div class="tip-card">
        <h4>📌 Important Note</h4>
        <p>
            This prediction is generated by a machine learning model and should
            be used only as an initial assessment. The final loan decision should
            also consider official financial verification, repayment capacity,
            documentation, and institutional policies.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    <p>🏦 <strong>LoanWise AI</strong> — Intelligent Loan Approval Prediction</p>
    <p>Built with Python • Pandas • Scikit-learn • Streamlit</p>
</div>
""", unsafe_allow_html=True)