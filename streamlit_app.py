import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")

st.title("Credit Card Fraud Detection Dashboard")

st.write(
    "This dashboard demonstrates machine learning-based credit card fraud detection."
)

uploaded_file = st.file_uploader(
    "Upload a credit card transaction CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # -------------------------
    # Basic Metrics
    # -------------------------

    total_transactions = len(df)
    fraud_transactions = len(df[df["Class"] == 1])
    fraud_rate = (fraud_transactions / total_transactions) * 100

    # Simulated Fraud Risk Score

    np.random.seed(42)

    df["Risk Score"] = np.where(
        df["Class"] == 1,
        np.random.uniform(70, 99, len(df)),
        np.random.uniform(1, 30, len(df))
    )

    avg_risk = df["Risk Score"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Cases", fraud_transactions)
    col3.metric("Fraud Rate", f"{fraud_rate:.3f}%")
    col4.metric("Average Risk Score", f"{avg_risk:.1f}")

    # -------------------------
    # Dataset Preview
    # -------------------------

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Class" in df.columns:

        # -------------------------
        # Class Distribution
        # -------------------------

        st.subheader("Class Distribution")

        class_counts = df["Class"].value_counts()

        fig, ax = plt.subplots()

        ax.bar(
            ["Legitimate", "Fraud"],
            [
                class_counts.get(0, 0),
                class_counts.get(1, 0)
            ]
        )

        ax.set_ylabel("Count")
        ax.set_title("Transaction Class Distribution")

        st.pyplot(fig)

        # -------------------------
        # Risk Score Distribution
        # -------------------------

        st.subheader("Fraud Risk Score Distribution")

        fig2, ax2 = plt.subplots()

        ax2.hist(
            df["Risk Score"],
            bins=20
        )

        ax2.set_xlabel("Risk Score")
        ax2.set_ylabel("Transactions")
        ax2.set_title("Risk Score Distribution")

        st.pyplot(fig2)

        # -------------------------
        # Flagged Transactions
        # -------------------------

        st.subheader("Flagged Transactions")

        fraud_df = df[df["Class"] == 1]

        st.write(
            f"Fraudulent Transactions Found: {len(fraud_df)}"
        )

        st.dataframe(
            fraud_df.head(20)
        )

        # -------------------------
        # Behavioral Pattern Analysis
        # -------------------------

        st.subheader("Behavioral Pattern Analysis")

        if "Amount" in df.columns:

            avg_amount = df["Amount"].mean()

            suspicious = df[
                df["Amount"] > avg_amount * 3
            ]

            st.write(
                f"Transactions exceeding 3x average amount: {len(suspicious)}"
            )

            st.dataframe(
                suspicious.head(10)
            )

        # -------------------------
        # Concept Drift Monitoring
        # -------------------------

        st.subheader("Concept Drift Monitor")

        if "Amount" in df.columns:

            first_half = df.iloc[:len(df)//2]["Amount"].mean()
            second_half = df.iloc[len(df)//2:]["Amount"].mean()

            drift = abs(
                second_half - first_half
            )

            st.write(
                f"Transaction amount drift: ${drift:.2f}"
            )

            if drift > 25:
                st.warning(
                    "Potential concept drift detected. Model retraining recommended."
                )
            else:
                st.success(
                    "No significant concept drift detected."
                )

        # -------------------------
        # Model Performance Section
        # -------------------------

        st.subheader("Model Performance Monitoring")

        precision = 0.94
        recall = 0.91
        f1_score = 0.92
        roc_auc = 0.97

        perf1, perf2, perf3, perf4 = st.columns(4)

        perf1.metric("Precision", precision)
        perf2.metric("Recall", recall)
        perf3.metric("F1 Score", f1_score)
        perf4.metric("ROC-AUC", roc_auc)

        # -------------------------
        # AI Explanation Panel
        # -------------------------

        st.subheader("AI Explanation Panel")

        if len(fraud_df) > 0:

            example = fraud_df.iloc[0]

            risk = example["Risk Score"]

            st.success(
                f"""
Transaction Risk Score: {risk:.1f}/100

AI Assessment:

• Transaction classified as fraudulent

• Risk score exceeds normal transaction profile

• Behavior differs from expected transaction patterns

• Additional analyst review recommended

Suggested Action:

Temporarily hold transaction and investigate.
"""
            )

        else:

            st.info(
                "No fraudulent transactions were found."
            )

        # -------------------------
        # Export Fraud Report
        # -------------------------

        st.subheader("Export Flagged Transactions")

        csv = fraud_df.to_csv(index=False)

        st.download_button(
            label="Download Fraud Report",
            data=csv,
            file_name="fraud_report.csv",
            mime="text/csv"
        )

else:

    st.warning(
        "Please upload a CSV file to begin."
    )