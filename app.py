import streamlit as st
import pandas as pd
import numpy as np

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Data Quality Monitoring System",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Intelligent Data Quality Monitoring System")
st.markdown("""
This system monitors financial transaction datasets by performing:

- Data Profiling
- Missing Value Detection
- Duplicate Detection
- Data Quality Scoring
- Z-Score Anomaly Detection
- Isolation Forest Anomaly Detection
""")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Financial Transaction Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully.")

    # --------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------

    st.header("📄 Dataset Preview")

    st.dataframe(df.head())

    # --------------------------------------------------
    # DATA PROFILING
    # --------------------------------------------------

    st.header("📌 Data Profiling")

    total_records = df.shape[0]
    total_columns = df.shape[1]

    missing_values = df.isnull().sum().sum()
    duplicate_records = df.duplicated().sum()

    total_cells = total_records * total_columns

    quality_score = (
        (total_cells - missing_values)
        / total_cells
    ) * 100

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Records",
        total_records
    )

    col2.metric(
        "Columns",
        total_columns
    )

    col3.metric(
        "Missing Values",
        int(missing_values)
    )

    col4.metric(
        "Duplicates",
        int(duplicate_records)
    )

    col5.metric(
        "Quality Score",
        f"{quality_score:.2f}%"
    )

    # --------------------------------------------------
    # DATA QUALITY STATUS
    # --------------------------------------------------

    st.header("🟢 Data Quality Status")

    if quality_score >= 95:
        st.success("Excellent Data Quality")

    elif quality_score >= 85:
        st.warning("Moderate Data Quality")

    else:
        st.error("Poor Data Quality")

    # --------------------------------------------------
    # MISSING VALUE REPORT
    # --------------------------------------------------

    st.header("🔍 Missing Value Analysis")

    missing_report = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(missing_report)

    # --------------------------------------------------
    # DUPLICATE RECORD ANALYSIS
    # --------------------------------------------------

    st.header("📑 Duplicate Record Analysis")

    st.write(
        f"Total Duplicate Records Found: {duplicate_records}"
    )

    if duplicate_records > 0:

        st.dataframe(
            df[df.duplicated()]
        )

    # --------------------------------------------------
    # NUMERIC COLUMN DETECTION
    # --------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # --------------------------------------------------
    # SELECT COLUMN FOR ANOMALY DETECTION
    # --------------------------------------------------

    if len(numeric_columns) > 0:

        st.header("🤖 Anomaly Detection")

        selected_column = st.selectbox(
            "Select Numeric Column",
            numeric_columns
        )

        # ----------------------------------------------
        # Z-SCORE DETECTION
        # ----------------------------------------------

        st.subheader("Z-Score Detection")

        temp_df = df.copy()

        temp_df[selected_column] = pd.to_numeric(
            temp_df[selected_column],
            errors="coerce"
        )

        temp_df = temp_df.dropna(
            subset=[selected_column]
        )

        temp_df["Z_Score"] = zscore(
            temp_df[selected_column]
        )

        zscore_anomalies = temp_df[
            abs(temp_df["Z_Score"]) > 3
        ]

        st.write(
            f"Anomalies Detected: {len(zscore_anomalies)}"
        )

        st.dataframe(
            zscore_anomalies
        )

        # ----------------------------------------------
        # ISOLATION FOREST
        # ----------------------------------------------

        st.subheader("Isolation Forest Detection")

        model = IsolationForest(
            contamination=0.02,
            random_state=42
        )

        temp_df["Anomaly"] = model.fit_predict(
            temp_df[[selected_column]]
        )

        isolation_anomalies = temp_df[
            temp_df["Anomaly"] == -1
        ]

        st.write(
            f"Anomalies Detected: {len(isolation_anomalies)}"
        )

        st.dataframe(
            isolation_anomalies
        )

    else:

        st.warning(
            "No numeric columns available for anomaly detection."
        )

    # --------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------

    st.header("📊 Data Structure")

    structure_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(structure_df)




    # --------------------------------------------------
    # DATA VALIDATION
    # --------------------------------------------------

    st.header("✅ Data Validation")

    validation_df = df.copy()

    # Convert amount columns to numeric
    amount_columns = [
        " WITHDRAWAL AMT ",
        " DEPOSIT AMT ",
        "BALANCE AMT"
    ]

    for col in amount_columns:
        if col in validation_df.columns:
            validation_df[col] = pd.to_numeric(
                validation_df[col],
                errors="coerce"
            )

    # -------------------------------
    # Account Number Validation
    # -------------------------------

    if "Account No" in validation_df.columns:

        invalid_account = validation_df[
            validation_df["Account No"].isnull()
        ]

        st.subheader("Account Number Validation")

        st.write(
            f"Invalid Account Numbers: {len(invalid_account)}"
        )

        if len(invalid_account) > 0:
            st.dataframe(invalid_account)
        else:
            st.success("No invalid account numbers found.")

    # -------------------------------
    # Transaction Date Validation
    # -------------------------------

    if "DATE" in validation_df.columns:

        validation_df["DATE"] = pd.to_datetime(
            validation_df["DATE"],
            errors="coerce"
        )

        invalid_dates = validation_df[
            validation_df["DATE"].isnull()
        ]

        st.subheader("Transaction Date Validation")

        st.write(
            f"Invalid Dates: {len(invalid_dates)}"
        )

        if len(invalid_dates) > 0:
            st.dataframe(invalid_dates)
        else:
            st.success("No invalid dates found.")

    # -------------------------------
    # Withdrawal Validation
    # -------------------------------

    withdraw_col = " WITHDRAWAL AMT "

    if withdraw_col in validation_df.columns:

        invalid_withdrawal = validation_df[
            validation_df[withdraw_col] < 0
        ]

        st.subheader("Withdrawal Amount Validation")

        st.write(
            f"Negative Withdrawals: {len(invalid_withdrawal)}"
        )

        if len(invalid_withdrawal) > 0:
            st.dataframe(invalid_withdrawal)
        else:
            st.success("No negative withdrawal amounts found.")

    # -------------------------------
    # Deposit Validation
    # -------------------------------

    deposit_col = " DEPOSIT AMT "

    if deposit_col in validation_df.columns:

        invalid_deposit = validation_df[
            validation_df[deposit_col] < 0
        ]

        st.subheader("Deposit Amount Validation")

        st.write(
            f"Negative Deposits: {len(invalid_deposit)}"
        )

        if len(invalid_deposit) > 0:
            st.dataframe(invalid_deposit)
        else:
            st.success("No negative deposit amounts found.")

    # -------------------------------
    # Balance Validation
    # -------------------------------

    balance_col = "BALANCE AMT"

    if balance_col in validation_df.columns:

        invalid_balance = validation_df[
            validation_df[balance_col] < 0
        ]

        st.subheader("Balance Validation")

        st.write(
            f"Negative Balances: {len(invalid_balance)}"
        )

        if len(invalid_balance) > 0:
            st.dataframe(invalid_balance)
        else:
            st.success("No negative balances found.")

    # --------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------

    st.header("⬇ Download Clean Dataset")

    csv = validation_df.to_csv(index=False)

    st.download_button(
        label="Download Dataset",
        data=csv,
        file_name="data_quality_report.csv",
        mime="text/csv"
    )

 
    st.info("Please upload a CSV file to start analysis.")
