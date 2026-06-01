import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import shap
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("data/application_train.csv")
model = joblib.load("models/credit_risk_model.pkl")

# =====================================================
# SESSION STATE
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.hero {
    padding: 40px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #0f4c81,
        #1e88e5
    );
    color: white;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 48px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 20px;
}

.metric-card {
    background: white;
    padding: 35px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.metric-title {
    color: #666;
    font-size: 18px;
}

.metric-value {
    color: #0f4c81;
    font-size: 40px;
    font-weight: bold;
    margin-top: 10px;
}

.section-title {
    font-size: 30px;
    font-weight: bold;
    margin-top: 30px;
    margin-bottom: 20px;
}

.nav-card {
    background: white;
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    border-left: 6px solid #1e88e5;
    transition: 0.3s ease;
}

.nav-card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.12);
}

.nav-title {
    font-size: 26px;
    font-weight: bold;
    color: #0f4c81;
}

.nav-desc {
    font-size: 16px;
    color: #666;
    margin-top: 10px;
}

.big-button button {
    height: 65px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HOME PAGE
# =====================================================

if st.session_state.page == "home":

    st.markdown("""
    <div class="hero">
        <h1>💳 Credit Risk Intelligence Platform</h1>
        <p>
        Enterprise-grade credit risk assessment powered by
        Machine Learning, Explainable AI, and Interactive Analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    total_records = len(df)
    total_features = len(df.columns)
    default_cases = int(df["TARGET"].sum())
    default_rate = (default_cases / total_records) * 100

    # KPI ROW 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📄 Applications</div>
            <div class="metric-value">{total_records:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📊 Features</div>
            <div class="metric-value">{total_features}</div>
        </div>
        """, unsafe_allow_html=True)

    # KPI ROW 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚠️ Default Cases</div>
            <div class="metric-value">{default_cases:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📈 Default Rate</div>
            <div class="metric-value">{default_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        "<div class='section-title'>🚀 Quick Access</div>",
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown("""
        <div class="nav-card">
            <div class="nav-title">📊 Analytics Dashboard</div>
            <div class="nav-desc">
                Explore trends, distributions and business insights.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Analytics Dashboard",
            use_container_width=True,
            key="analytics"
        ):
            st.session_state.page = "dashboard"
            st.rerun()

        st.markdown("""
        <div class="nav-card">
            <div class="nav-title">⚠️ Risk Prediction</div>
            <div class="nav-desc">
                Predict customer loan default probability.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Risk Prediction",
            use_container_width=True,
            key="prediction"
        ):
            st.session_state.page = "prediction"
            st.rerun()

    with right:

        st.markdown("""
        <div class="nav-card">
            <div class="nav-title">🔍 Explainable AI</div>
            <div class="nav-desc">
                Understand model decisions using SHAP.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Explainability",
            use_container_width=True,
            key="explain"
        ):
            st.session_state.page = "explain"
            st.rerun()

        st.markdown("""
        <div class="nav-card">
            <div class="nav-title">🤖 AI Assistant</div>
            <div class="nav-desc">
                Ask questions about your dataset naturally.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open AI Assistant",
            use_container_width=True,
            key="chatbot"
        ):
            st.session_state.page = "chatbot"
            st.rerun()

        st.markdown("---")

        st.info(
            f"""
            📌 Dataset contains {total_records:,} loan applications
            with {total_features} financial attributes.
            Current observed default rate is
            {default_rate:.2f}%.
            """
        )

# =====================================================
# ANALYTICS PAGE
# =====================================================

elif st.session_state.page == "dashboard":

    st.title("📊 Analytics Dashboard")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("---")

    # ====================================
    # KPI SECTION
    # ====================================

    total_records = len(df)
    default_cases = int(df["TARGET"].sum())

    avg_income = df["AMT_INCOME_TOTAL"].mean()
    avg_credit = df["AMT_CREDIT"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Applications",
        f"{total_records:,}"
    )

    c2.metric(
        "Defaults",
        f"{default_cases:,}"
    )

    c3.metric(
        "Avg Income",
        f"{avg_income:,.0f}"
    )

    c4.metric(
        "Avg Credit",
        f"{avg_credit:,.0f}"
    )

    st.markdown("---")

    # ====================================
    # PIE CHART
    # ====================================

    left, right = st.columns(2)

    with left:

        st.subheader("Loan Default Distribution")

        target_counts = df["TARGET"].value_counts()

        pie_fig = px.pie(
            values=target_counts.values,
            names=["No Default", "Default"],
            hole=0.5,
            title="Default vs Non-Default"
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )

    # ====================================
    # INCOME DISTRIBUTION
    # ====================================

    with right:

        st.subheader("Income Distribution")

        income_fig = px.histogram(
            df,
            x="AMT_INCOME_TOTAL",
            nbins=50,
            title="Applicant Income Distribution"
        )

        st.plotly_chart(
            income_fig,
            use_container_width=True
        )

    st.markdown("---")

    # ====================================
    # MISSING VALUES
    # ====================================

    st.subheader("Top Missing Columns")

    missing = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    missing_fig = px.bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        title="Missing Values Analysis"
    )

    st.plotly_chart(
        missing_fig,
        use_container_width=True
    )

    missing_percent = (
        df.isnull().sum().sum()
        /
        (df.shape[0] * df.shape[1])
    ) * 100

    st.metric(
        "Dataset Health",
        f"{100-missing_percent:.2f}%"
    )

# =====================================================
# PREDICTION PAGE
# =====================================================

elif st.session_state.page == "prediction":

    st.title("⚠️ Credit Risk Prediction")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("---")

    st.subheader("Select Applicant")

    applicant_id = st.number_input(
        "Applicant Row Number",
        min_value=0,
        max_value=len(df)-1,
        value=0
    )

    applicant = df.iloc[applicant_id]

    st.markdown("### Applicant Overview")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Income",
            f"{applicant['AMT_INCOME_TOTAL']:,.0f}"
        )

        st.metric(
            "Credit Amount",
            f"{applicant['AMT_CREDIT']:,.0f}"
        )

    with c2:

        st.metric(
            "Annuity",
            f"{applicant['AMT_ANNUITY']:,.0f}"
            if pd.notnull(applicant['AMT_ANNUITY'])
            else "N/A"
        )

        st.metric(
            "Goods Price",
            f"{applicant['AMT_GOODS_PRICE']:,.0f}"
            if pd.notnull(applicant['AMT_GOODS_PRICE'])
            else "N/A"
        )

    st.markdown("---")

    if st.button(
        "🔍 Predict Credit Risk",
        use_container_width=True
    ):

        X = df.drop(
            columns=["TARGET", "SK_ID_CURR"]
        )

        sample = X.iloc[[applicant_id]]

        probability = model.predict_proba(
            sample
        )[0][1]

        if probability < 0.30:

            risk = "🟢 LOW RISK"

        elif probability < 0.70:

            risk = "🟡 MEDIUM RISK"

        else:

            risk = "🔴 HIGH RISK"

        st.success(
            f"Default Probability: {probability:.2%}"
        )

        if probability < 0.30:
            st.success(f"🟢 LOW RISK ({probability:.2%})")

        elif probability < 0.70:
            st.warning(f"🟡 MEDIUM RISK ({probability:.2%})")

        else:
            st.error(f"🔴 HIGH RISK ({probability:.2%})")

# =====================================================
# EXPLAINABILITY PAGE
# =====================================================

elif st.session_state.page == "explain":

    st.title("🔍 Explainable AI")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("---")

    st.subheader("Model Transparency Dashboard")

    st.write(
        """
        SHAP (SHapley Additive Explanations)
        helps explain how features influence
        model predictions.
        """
    )

    X = df.drop(
        columns=["TARGET", "SK_ID_CURR"]
    )

    st.markdown("---")

    st.subheader("Select Applicant")

    applicant_id = st.number_input(
        "Applicant Index",
        min_value=0,
        max_value=len(X)-1,
        value=0,
        key="shap_applicant"
    )

    sample = X.iloc[[applicant_id]]
    st.markdown("---")

    st.subheader("Global Feature Importance")

    model_step = model.named_steps["model"]

    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    importances = model.named_steps["model"].feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.markdown("---")

    st.subheader("Selected Applicant")

    st.dataframe(
        sample.T.head(20)
    )

# =====================================================
# CHATBOT PAGE
# =====================================================

elif st.session_state.page == "chatbot":

    st.title("🤖 Credit Risk Assistant")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("---")

    st.write(
        """
        Ask questions about the dataset.

        Examples:
        • What is the average income?
        • How many defaults are there?
        • What is the default rate?
        • How many applications?
        """
    )

    question = st.text_input(
        "Ask your question"
    )

    if question:

        q = question.lower()

        if "average income" in q:

            avg_income = (
                df["AMT_INCOME_TOTAL"]
                .mean()
            )

            st.success(
                f"Average Income: {avg_income:,.2f}"
            )

        elif "default rate" in q:

            default_rate = (
                df["TARGET"].mean()
            ) * 100

            st.success(
                f"Default Rate: {default_rate:.2f}%"
            )

        elif "defaults" in q:

            defaults = (
                df["TARGET"]
                .sum()
            )

            st.success(
                f"Total Defaults: {defaults:,}"
            )

        elif "applications" in q:

            st.success(
                f"Total Applications: {len(df):,}"
            )

        else:

            st.warning(
                "Question not recognized yet."
            )

st.markdown("---")

st.caption(
    "Credit Risk Intelligence Platform | "
    "Built with Streamlit, LightGBM, SHAP and Plotly"
)