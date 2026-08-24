import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

import plotly.express as px
import plotly.graph_objects as go



# PAGE CONFIGURATION


st.set_page_config(
    page_title="ClaimShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)



# CUSTOM CSS


st.markdown("""
<style>

.main {
    background-color: #f6f8fb;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    color: #64748b;
    margin-bottom: 30px;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #2563eb;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}

.risk-high {
    color: white;
    background-color: #7F0000;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #dc2626;
}

.risk-medium {
    color: white;
    background-color: #E88504;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #d97706;
}

.risk-low {
    color: white;
    background-color: #043927;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #16a34a;
}

.info-box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)



# LOAD DATA


@st.cache_data
def load_data():

    df = pd.read_csv("ClaimShield_AI_Evaluation_Results.csv")
    raw_df = pd.read_csv("evaluation_dataset.csv")

    return df, raw_df


@st.cache_resource
def load_artifacts():

    with open("claimshield_features.pkl", "rb") as file:
        features = pickle.load(file)

    with open("claimshield_threshold.pkl", "rb") as file:
        threshold = pickle.load(file)

    return features, threshold


df, raw_df = load_data()
features, threshold = load_artifacts()



# HELPER FUNCTIONS


def get_risk_level(probability):

    if probability >= 0.80:
        return "Critical"

    elif probability >= threshold:
        return "High"

    elif probability >= 0.30:
        return "Medium"

    else:
        return "Low"


def get_recommended_action(probability):

    if probability >= 0.80:
        return "ESCALATE FOR IMMEDIATE INVESTIGATION"

    elif probability >= threshold:
        return "REVIEW CLAIM"

    elif probability >= 0.30:
        return "ADDITIONAL VERIFICATION"

    else:
        return "PROCESS NORMALLY"


def create_features(
        age,
        gender,
        region,
        policy_type,
        premium_amount,
        coverage_amount,
        claim_amount,
        claim_type,
        credit_score
):

    claim_coverage_ratio = (
        claim_amount / coverage_amount
        if coverage_amount != 0 else 0
    )

    claim_premium_ratio = (
        claim_amount / premium_amount
        if premium_amount != 0 else 0
    )

    coverage_premium_ratio = (
        coverage_amount / premium_amount
        if premium_amount != 0 else 0
    )

    remaining_coverage = coverage_amount - claim_amount

    young_customer = int(age < 30)

    senior_customer = int(age >= 60)

    low_credit_score = int(credit_score < 500)

    high_claim = int(
        claim_coverage_ratio >= 0.70
    )

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Region": region,
        "Policy_Type": policy_type,
        "Premium_Amount": premium_amount,
        "Coverage_Amount": coverage_amount,
        "Claim_Amount": claim_amount,
        "Claim_Type": claim_type,
        "Credit_Score": credit_score,
        "Claim_Coverage_Ratio": claim_coverage_ratio,
        "Claim_Premium_Ratio": claim_premium_ratio,
        "Coverage_Premium_Ratio": coverage_premium_ratio,
        "Remaining_Coverage": remaining_coverage,
        "Young_Customer": young_customer,
        "Senior_Customer": senior_customer,
        "Low_Credit_Score": low_credit_score,
        "High_Claim": high_claim
    }])

    return input_data



# SIDEBAR


with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/96/shield.png",
        width=80
    )

    st.title("ClaimShield AI")

    st.caption(
        "Insurance Claim Fraud Intelligence Platform"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Executive Dashboard",
            "🔍 Analyze New Claim",
            "📊 Portfolio Analytics",
            "🚨 Investigation Queue",
            "📄 Claim Explorer",
            "🤖 Model Information"
        ]
    )

    st.divider()

    st.markdown("### System Status")

    st.success("● Model System Ready")

    st.info(
        f"Decision Threshold: {threshold:.3f}"
    )

    st.caption(
        "Powered by XGBoost Fraud Detection"
    )



# PAGE 1: EXECUTIVE DASHBOARD


if page == "🏠 Executive Dashboard":

    st.markdown(
        '<p class="main-title" style="color: white">🛡️ ClaimShield AI</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">'
        'AI-Powered Insurance Claim Fraud Detection and Risk Intelligence Platform'
        '</p>',
        unsafe_allow_html=True
    )

    total_claims = len(df)

    fraud_claims = (
        df["Fraud_Prediction"] == "Fraudulent"
    ).sum()

    genuine_claims = (
        df["Fraud_Prediction"] == "Genuine"
    ).sum()

    fraud_rate = (
        fraud_claims / total_claims
    ) * 100

    avg_probability = (
        df["Fraud_Probability"].mean()
    ) * 100

    total_exposure = (
        df["Potential_Financial_Loss"].sum()
    )


    
    # KPI SECTION
    

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Claims Analyzed",
        f"{total_claims:,}"
    )

    col2.metric(
        "Fraudulent Claims",
        f"{fraud_claims:,}",
        f"{fraud_rate:.1f}% of portfolio"
    )

    col3.metric(
        "Genuine Claims",
        f"{genuine_claims:,}"
    )

    col4.metric(
        "Potential Financial Exposure",
        f"₹{total_exposure:,.0f}"
    )


    st.divider()


    
    # SECONDARY METRICS
    

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Fraud Probability",
        f"{avg_probability:.2f}%"
    )

    review_claims = (
        df["Recommended_Action"] == "REVIEW"
    ).sum()

    col2.metric(
        "Claims Requiring Review",
        review_claims
    )

    high_risk = (
        df["Fraud_Probability"] >= 0.80
    ).sum()

    col3.metric(
        "Critical Risk Claims",
        high_risk
    )


    st.divider()


    
    # EXECUTIVE INSIGHTS
    

    st.subheader("📌 Executive Risk Summary")

    col1, col2 = st.columns([1.3, 1])

    with col1:

        fraud_distribution = (
            df["Fraud_Prediction"]
            .value_counts()
            .reset_index()
        )

        fraud_distribution.columns = [
            "Prediction",
            "Count"
        ]

        fig = px.pie(
            fraud_distribution,
            names="Prediction",
            values="Count",
            hole=0.55,
            title="Fraud Detection Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        top_claims = (
            df.sort_values(
                "Potential_Financial_Loss",
                ascending=False
            )
            .head(5)
        )

        st.markdown(
            "### 💰 Highest Financial Exposure"
        )

        st.dataframe(
            top_claims[
                [
                    "Claim_ID",
                    "Fraud_Probability",
                    "Fraud_Prediction",
                    "Potential_Financial_Loss"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


    
    # BUSINESS INTERPRETATION
    

    st.subheader("🧠 What This Means for the Business")

    st.markdown(f"""
    <div class="info-box" style="background: transparent">

    <b>Portfolio Overview:</b> ClaimShield AI has analyzed
    <b>{total_claims:,} insurance claims</b>.

    <br><br>

    <b>Fraud Risk:</b> The system identified
    <b>{fraud_claims:,} claims</b> as potentially fraudulent
    using the trained machine learning model and optimized
    decision threshold.

    <br><br>

    <b>Financial Protection:</b> Claims currently represent
    approximately <b>₹{total_exposure:,.0f}</b> in estimated
    potential financial exposure.

    <br><br>

    <b>Operational Action:</b> High-risk claims should be
    prioritized for investigation so that the insurance team
    can focus resources on cases with the greatest potential
    financial impact.

    </div>
    """, unsafe_allow_html=True)



# PAGE 2: ANALYZE NEW CLAIM


elif page == "🔍 Analyze New Claim":

    st.title("🔍 Analyze a New Insurance Claim")

    st.write(
        "Enter claim information below. ClaimShield AI will "
        "prepare the required engineered features and use the "
        "trained model to estimate fraud risk."
    )

    st.divider()


    
    # CUSTOMER DETAILS
    

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col3:
        region = st.selectbox(
            "Region",
            sorted(df["Region"].dropna().unique())
        )


    
    # POLICY DETAILS
    

    st.subheader("📋 Policy Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        policy_type = st.selectbox(
            "Policy Type",
            sorted(df["Policy_Type"].dropna().unique())
        )

    with col2:
        premium_amount = st.number_input(
            "Premium Amount",
            min_value=0.0,
            value=5000.0
        )

    with col3:
        coverage_amount = st.number_input(
            "Coverage Amount",
            min_value=1.0,
            value=50000.0
        )


    
    # CLAIM DETAILS
    

    st.subheader("📄 Claim Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        claim_amount = st.number_input(
            "Claim Amount",
            min_value=0.0,
            value=15000.0
        )

    with col2:
        claim_type = st.selectbox(
            "Claim Type",
            sorted(df["Claim_Type"].dropna().unique())
        )

    with col3:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )


    st.divider()


    if st.button(
        "🛡️ Analyze Fraud Risk",
        type="primary",
        use_container_width=True
    ):

        input_df = create_features(
            age,
            gender,
            region,
            policy_type,
            premium_amount,
            coverage_amount,
            claim_amount,
            claim_type,
            credit_score
        )

        st.session_state["input_df"] = input_df


        
        # MODEL PREDICTION
        # ----------------
        # The prediction code below uses the saved model.
        # If your environment has the same dependency
        # versions used to train the model, uncomment it.
       

        try:

            with open(
                "claimshield_final_xgboost.pkl",
                "rb"
            ) as file:

                model = pickle.load(file)

            probability = model.predict_proba(
                input_df[features]
            )[0][1]

        except Exception as e:

            # Fallback demonstration mode
            # Remove when model loads successfully

            probability = min(
                0.95,
                max(
                    0.05,
                    (
                        (claim_amount / coverage_amount) * 0.35
                        +
                        (1 - (credit_score / 900)) * 0.25
                        +
                        (claim_amount / premium_amount / 10) * 0.20
                    )
                )
            )

            st.warning(
                "Model artifact could not be loaded in the "
                "current Python environment. Demo risk calculation "
                "is being displayed. Ensure the Streamlit environment "
                "uses the same package versions as the training environment."
            )


        prediction = (
            "Fraudulent"
            if probability >= threshold
            else "Genuine"
        )

        risk_level = get_risk_level(
            probability
        )

        action = get_recommended_action(
            probability
        )

        potential_loss = (
            probability * claim_amount
        )


        
        # RESULTS
        

        st.success(
            "Claim analysis completed successfully."
        )

        st.divider()

        st.subheader(
            "🧠 AI Risk Assessment Result"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        col2.metric(
            "AI Prediction",
            prediction
        )

        col3.metric(
            "Risk Level",
            risk_level
        )

        col4.metric(
            "Potential Exposure",
            f"₹{potential_loss:,.2f}"
        )


        
        # RISK MESSAGE
        

        if risk_level == "Critical":

            st.markdown(
                f"""
                <div class="risk-high">

                <h3>🚨 CRITICAL FRAUD RISK</h3>

                The model has assigned a fraud probability of
                <b>{probability * 100:.2f}%</b>.

                This claim should be immediately escalated to
                the fraud investigation team.

                <br><br>

                <b>Recommended Action:</b> {action}

                </div>
                """,
                unsafe_allow_html=True
            )

        elif risk_level in ["High", "Medium"]:

            st.markdown(
                f"""
                <div class="risk-medium">

                <h3>⚠️ CLAIM REQUIRES ATTENTION</h3>

                The claim has a fraud probability of
                <b>{probability * 100:.2f}%</b>.

                The system recommends additional verification
                before final settlement.

                <br><br>

                <b>Recommended Action:</b> {action}

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="risk-low">

                <h3>✅ LOW FRAUD RISK</h3>

                The model estimates a relatively low probability
                of fraudulent activity.

                <br><br>

                <b>Recommended Action:</b> {action}

                </div>
                """,
                unsafe_allow_html=True
            )


        
        # FEATURE ANALYSIS
        

        st.subheader(
            "📊 Claim Risk Indicators"
        )

        indicators = pd.DataFrame({
            "Indicator": [
                "Claim / Coverage Ratio",
                "Claim / Premium Ratio",
                "Coverage / Premium Ratio",
                "Remaining Coverage",
                "Young Customer",
                "Senior Customer",
                "Low Credit Score",
                "High Claim Indicator"
            ],
            "Value": [
                input_df["Claim_Coverage_Ratio"].iloc[0],
                input_df["Claim_Premium_Ratio"].iloc[0],
                input_df["Coverage_Premium_Ratio"].iloc[0],
                input_df["Remaining_Coverage"].iloc[0],
                input_df["Young_Customer"].iloc[0],
                input_df["Senior_Customer"].iloc[0],
                input_df["Low_Credit_Score"].iloc[0],
                input_df["High_Claim"].iloc[0]
            ]
        })

        st.dataframe(
            indicators,
            use_container_width=True,
            hide_index=True
        )

        with st.expander(
            "🔎 View Full Feature Engineering Details"
        ):

            st.dataframe(
                input_df.T.rename(
                    columns={0: "Value"}
                ),
                use_container_width=True
            )



# PAGE 3: PORTFOLIO ANALYTICS


elif page == "📊 Portfolio Analytics":

    st.title("📊 Insurance Fraud Portfolio Analytics")

    st.write(
        "Explore patterns across the evaluated insurance claim portfolio."
    )


    
    # FILTERS
    

    st.sidebar.markdown(
        "### Portfolio Filters"
    )

    selected_region = st.sidebar.multiselect(
        "Region",
        df["Region"].unique(),
        default=df["Region"].unique()
    )

    selected_policy = st.sidebar.multiselect(
        "Policy Type",
        df["Policy_Type"].unique(),
        default=df["Policy_Type"].unique()
    )

    filtered_df = df[
        (df["Region"].isin(selected_region))
        &
        (df["Policy_Type"].isin(selected_policy))
    ]


    
    # FRAUD BY REGION
    

    col1, col2 = st.columns(2)

    with col1:

        region_fraud = pd.crosstab(
            filtered_df["Region"],
            filtered_df["Fraud_Prediction"]
        ).reset_index()

        fig = px.bar(
            region_fraud,
            x="Region",
            y=[
                col for col in region_fraud.columns
                if col != "Region"
            ],
            barmode="group",
            title="Fraud Prediction by Region"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    
    # POLICY TYPE
    

    with col2:

        policy_fraud = (
            filtered_df.groupby(
                "Policy_Type"
            )["Fraud_Probability"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            policy_fraud,
            x="Policy_Type",
            y="Fraud_Probability",
            title="Average Fraud Probability by Policy Type"
        )

        fig.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    
    # CLAIM AMOUNT VS RISK
    

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            filtered_df,
            x="Claim_Amount",
            y="Fraud_Probability",
            color="Fraud_Prediction",
            hover_data=[
                "Claim_ID",
                "Credit_Score",
                "Potential_Financial_Loss"
            ],
            title="Claim Amount vs Fraud Probability"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    
    # FINANCIAL EXPOSURE
    

    with col2:

        claim_type_exposure = (
            filtered_df.groupby(
                "Claim_Type"
            )["Potential_Financial_Loss"]
            .sum()
            .reset_index()
            .sort_values(
                "Potential_Financial_Loss",
                ascending=False
            )
        )

        fig = px.bar(
            claim_type_exposure,
            x="Claim_Type",
            y="Potential_Financial_Loss",
            title="Potential Financial Exposure by Claim Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    
    # DISTRIBUTION
    

    st.subheader(
        "Fraud Probability Distribution"
    )

    fig = px.histogram(
        filtered_df,
        x="Fraud_Probability",
        nbins=30,
        color="Fraud_Prediction",
        title="Distribution of AI Fraud Risk Scores"
    )

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        annotation_text=f"Decision Threshold: {threshold:.3f}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )



# PAGE 4: INVESTIGATION QUEUE


elif page == "🚨 Investigation Queue":

    st.title(
        "🚨 Fraud Investigation Queue"
    )

    st.write(
        "Prioritized claims that may require investigation by the fraud and risk team."
    )


    queue_df = df.copy()

    min_probability = st.slider(
        "Minimum Fraud Probability",
        min_value=0.0,
        max_value=1.0,
        value=float(threshold),
        step=0.01
    )

    queue_df = queue_df[
        queue_df["Fraud_Probability"]
        >= min_probability
    ]

    queue_df = queue_df.sort_values(
        [
            "Fraud_Probability",
            "Potential_Financial_Loss"
        ],
        ascending=False
    )


    st.metric(
        "Claims Currently in Investigation Queue",
        len(queue_df)
    )


    display_columns = [
        "Claim_ID",
        "Customer_ID",
        "Region",
        "Policy_Type",
        "Claim_Type",
        "Claim_Amount",
        "Fraud_Probability",
        "Fraud_Prediction",
        "Recommended_Action",
        "Potential_Financial_Loss"
    ]

    st.dataframe(
        queue_df[display_columns],
        use_container_width=True,
        hide_index=True
    )


    csv = queue_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Investigation Queue",
        csv,
        "claimshield_investigation_queue.csv",
        "text/csv"
    )



# PAGE 5: CLAIM EXPLORER


elif page == "📄 Claim Explorer":

    st.title(
        "📄 Individual Claim Explorer"
    )

    st.write(
        "Search an individual claim and review its complete risk profile."
    )

    claim_ids = df["Claim_ID"].unique()

    selected_claim = st.selectbox(
        "Select Claim ID",
        claim_ids
    )

    claim_data = df[
        df["Claim_ID"] == selected_claim
    ].iloc[0]


    
    # CLAIM SUMMARY
    

    st.subheader(
        f"Claim Profile: {selected_claim}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Fraud Probability",
        f"{claim_data['Fraud_Probability'] * 100:.2f}%"
    )

    col2.metric(
        "Prediction",
        claim_data["Fraud_Prediction"]
    )

    col3.metric(
        "Recommended Action",
        claim_data["Recommended_Action"]
    )

    col4.metric(
        "Potential Loss",
        f"₹{claim_data['Potential_Financial_Loss']:,.2f}"
    )


    
    # CUSTOMER DETAILS
    

    st.subheader(
        "👤 Customer & Policy Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        customer_details = pd.DataFrame({
            "Field": [
                "Customer ID",
                "Age",
                "Gender",
                "Credit Score",
                "Region"
            ],
            "Value": [
                claim_data["Customer_ID"],
                claim_data["Age"],
                claim_data["Gender"],
                claim_data["Credit_Score"],
                claim_data["Region"]
            ]
        })

        st.dataframe(
            customer_details,
            hide_index=True,
            use_container_width=True
        )


    with col2:

        policy_details = pd.DataFrame({
            "Field": [
                "Policy Number",
                "Policy Type",
                "Premium Amount",
                "Coverage Amount",
                "Remaining Coverage"
            ],
            "Value": [
                claim_data["Policy_Number"],
                claim_data["Policy_Type"],
                f"₹{claim_data['Premium_Amount']:,.2f}",
                f"₹{claim_data['Coverage_Amount']:,.2f}",
                f"₹{claim_data['Remaining_Coverage']:,.2f}"
            ]
        })

        st.dataframe(
            policy_details,
            hide_index=True,
            use_container_width=True
        )


    
    # RISK BREAKDOWN
    

    st.subheader(
        "⚠️ Risk Indicator Breakdown"
    )

    risk_indicators = pd.DataFrame({
        "Indicator": [
            "Fraud Risk Score",
            "Claim / Coverage Ratio",
            "Claim / Premium Ratio",
            "Low Credit Score Flag",
            "High Claim Flag"
        ],
        "Value": [
            claim_data["Fraud_Risk_Score"],
            claim_data["Claim_Coverage_Ratio"],
            claim_data["Claim_Premium_Ratio"],
            claim_data["Low_Credit_Score"],
            claim_data["High_Claim"]
        ]
    })

    st.dataframe(
        risk_indicators,
        hide_index=True,
        use_container_width=True
    )



# PAGE 6: MODEL INFORMATION


elif page == "🤖 Model Information":

    st.title(
        "🤖 ClaimShield AI Model Information"
    )

    st.markdown("""
    ### Model Purpose

    ClaimShield AI is designed to assist insurance companies
    in identifying potentially fraudulent claims.

    The system analyzes customer information, policy information,
    claim details, financial relationships and engineered risk
    indicators to estimate the probability that a claim may be
    fraudulent.
    """)


    st.divider()


    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "⚙️ Model Configuration"
        )

        st.info(
            "Machine Learning Model: XGBoost"
        )

        st.info(
            f"Decision Threshold: {threshold:.3f}"
        )

        st.info(
            f"Number of Input Features: {len(features)}"
        )


    with col2:

        st.subheader(
            "🎯 Decision Logic"
        )

        st.markdown(f"""

        **Fraud Probability ≥ {threshold:.3f}**

        → Classified as **Fraudulent**

        **Fraud Probability < {threshold:.3f}**

        → Classified as **Genuine**

        The decision threshold is used instead of relying
        only on the default 0.50 probability cutoff.
        """)


    st.divider()


    st.subheader(
        "🧩 Features Used by the Model"
    )

    feature_df = pd.DataFrame({
        "Feature Number": range(
            1,
            len(features) + 1
        ),
        "Feature Name": features
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "🔄 AI Processing Pipeline"
    )

    st.markdown("""

    **1. Claim Data Input**

    ↓

    **2. Data Preparation**

    ↓

    **3. Feature Engineering**

    ↓

    **4. XGBoost Fraud Model**

    ↓

    **5. Fraud Probability Calculation**

    ↓

    **6. Optimized Threshold Decision**

    ↓

    **7. Fraud / Genuine Classification**

    ↓

    **8. Recommended Business Action**

    """)



# FOOTER


st.divider()

st.caption(
    "ClaimShield AI • Insurance Fraud Detection & Risk Intelligence System"
)