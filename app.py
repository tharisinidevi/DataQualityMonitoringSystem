
import streamlit as st
import pandas as pd
import numpy as np
import re

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Data Quality Monitoring System",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Intelligent Data Quality Monitoring System")

st.markdown("""
This system monitors financial transaction datasets by performing:

- Data Profiling
- Missing Value Detection
- Duplicate Detection
- Data Validation
- Z-Score Anomaly Detection
- Isolation Forest Anomaly Detection
""")


# =========================================================
# NAVIGATION
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"


c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)


with c1:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"


with c2:
    if st.button("📊 Profiling", use_container_width=True):
        st.session_state.page = "Profiling"


with c3:
    if st.button("🔍 Missing", use_container_width=True):
        st.session_state.page = "Missing"


with c4:
    if st.button("📑 Duplicates", use_container_width=True):
        st.session_state.page = "Duplicates"


with c5:
    if st.button("✅ Validation", use_container_width=True):
        st.session_state.page = "Validation"


with c6:
    if st.button("🤖 Anomaly", use_container_width=True):
        st.session_state.page = "Anomaly"


with c7:
    if st.button("📈 Structure", use_container_width=True):
        st.session_state.page = "Structure"


with c8:
    if st.button("📥 Report", use_container_width=True):
        st.session_state.page = "Report"


st.divider()


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Financial Transaction Dataset",
    type=["csv"]
)


if uploaded_file is None:

    st.info("Please upload a CSV file.")

    st.stop()


# =========================================================
# READ CSV
# =========================================================

df = pd.read_csv(
    uploaded_file
)


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .astype(str)
    .str.replace("\xa0", " ", regex=False)
    .str.strip()
)


# =========================================================
# REPLACE EMPTY VALUES WITH NaN
# =========================================================

df = df.replace(
    r"^\s*$",
    np.nan,
    regex=True
)


# =========================================================
# AUTOMATIC NUMERIC DETECTION
# =========================================================

numeric_detection_results = []


for col in df.columns:

    # -----------------------------------------------------
    # Skip columns that are already numeric
    # -----------------------------------------------------

    if pd.api.types.is_numeric_dtype(df[col]):

        numeric_detection_results.append({
            "Column": col,
            "Detected Type": "Numeric",
            "Numeric Ratio": "100%"
        })

        continue


    # -----------------------------------------------------
    # Work with text representation
    # -----------------------------------------------------

    original = df[col].copy()


    cleaned = (
        original
        .astype("string")
        .str.replace("\xa0", "", regex=False)
        .str.replace("\u2007", "", regex=False)
        .str.replace("\u202f", "", regex=False)
        .str.strip()
    )


    # -----------------------------------------------------
    # Remove commas
    # Example:
    #
    # 1,000,000.00
    #
    # becomes:
    #
    # 1000000.00
    # -----------------------------------------------------

    cleaned = cleaned.str.replace(
        ",",
        "",
        regex=False
    )


    # -----------------------------------------------------
    # Remove common currency symbols
    # -----------------------------------------------------

    cleaned = cleaned.str.replace(
        r"[$€£₹RM]",
        "",
        regex=True
    )


    # -----------------------------------------------------
    # Remove spaces inside numbers
    # -----------------------------------------------------

    cleaned = cleaned.str.replace(
        r"\s+",
        "",
        regex=True
    )


    # -----------------------------------------------------
    # Convert empty values to NaN
    # -----------------------------------------------------

    cleaned = cleaned.replace(
        ["", "nan", "None", "NULL", "null"],
        np.nan
    )


    # -----------------------------------------------------
    # Handle numbers written as:
    #
    # (1000)
    #
    # as:
    #
    # -1000
    # -----------------------------------------------------

    negative_parentheses = (
        cleaned
        .str.match(
            r"^\(.*\)$",
            na=False
        )
    )


    cleaned = cleaned.str.replace(
        r"^\((.*)\)$",
        r"-\1",
        regex=True
    )


    # -----------------------------------------------------
    # Check whether each value looks numeric
    #
    # Accepted examples:
    #
    # 1000
    # 1000.50
    # -1000
    # -1000.50
    # -----------------------------------------------------

    numeric_pattern = (
        r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$"
    )


    numeric_mask = (
        cleaned
        .str.fullmatch(
            numeric_pattern,
            na=False
        )
    )


    # -----------------------------------------------------
    # Count non-empty values
    # -----------------------------------------------------

    non_empty_count = (
        cleaned.notna()
        .sum()
    )


    numeric_count = (
        numeric_mask
        .sum()
    )


    # -----------------------------------------------------
    # Calculate numeric ratio
    # -----------------------------------------------------

    if non_empty_count > 0:

        numeric_ratio = (
            numeric_count
            / non_empty_count
        )

    else:

        numeric_ratio = 0


    # -----------------------------------------------------
    # CONVERT TO NUMERIC
    #
    # If at least 50% of non-empty values are numeric,
    # classify the column as numeric.
    #
    # This allows columns containing many blank cells,
    # such as withdrawal/deposit amounts.
    # -----------------------------------------------------

    if numeric_ratio >= 0.50:

        numeric_values = pd.to_numeric(
            cleaned,
            errors="coerce"
        )


        df[col] = numeric_values


        numeric_detection_results.append({

            "Column":
                col,

            "Detected Type":
                "Numeric",

            "Numeric Ratio":
                f"{numeric_ratio * 100:.2f}%"

        })

    else:

        numeric_detection_results.append({

            "Column":
                col,

            "Detected Type":
                "Text",

            "Numeric Ratio":
                f"{numeric_ratio * 100:.2f}%"

        })


# =========================================================
# SUCCESS MESSAGE
# =========================================================

st.success(
    "Dataset uploaded and automatically processed successfully."
)


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "Home":

    st.header("🏠 Home")

    st.markdown("""
    ### Welcome

    Welcome to the Intelligent Data Quality Monitoring System.

    This system automatically evaluates uploaded datasets using
    multiple data quality techniques.

    ### Available Functions

    📊 **Data Profiling**

    Provides an overview of the dataset including records,
    columns, missing values, duplicates, and quality score.

    🔍 **Missing Value Analysis**

    Identifies missing values in each column.

    📑 **Duplicate Detection**

    Identifies duplicate records.

    ✅ **Data Validation**

    Performs automatic validation of numerical and date fields.

    🤖 **Anomaly Detection**

    Uses Z-Score and Isolation Forest to identify potential
    anomalies in numerical data.

    📈 **Data Structure**

    Displays automatically detected data types.

    📥 **Data Quality Report**

    Generates a downloadable summary of the analysis.
    """)


# =========================================================
# DATA PROFILING
# =========================================================

elif st.session_state.page == "Profiling":

    st.header("📊 Data Profiling")


    total_records = df.shape[0]

    total_columns = df.shape[1]

    missing_values = (
        df.isnull()
        .sum()
        .sum()
    )

    duplicate_records = (
        df.duplicated()
        .sum()
    )

    total_cells = (
        total_records
        * total_columns
    )


    if total_cells > 0:

        quality_score = (
            (
                total_cells
                - missing_values
            )
            / total_cells
        ) * 100

    else:

        quality_score = 0


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


# =========================================================
# MISSING VALUE ANALYSIS
# =========================================================

elif st.session_state.page == "Missing":

    st.header("🔍 Missing Value Analysis")


    missing_report = pd.DataFrame({

        "Column":
            df.columns,

        "Missing Values":
            df.isnull()
            .sum()
            .values

    })


    st.dataframe(
        missing_report,
        use_container_width=True
    )


# =========================================================
# DUPLICATE DETECTION
# =========================================================

elif st.session_state.page == "Duplicates":

    st.header("📑 Duplicate Records")


    duplicate_records = (
        df.duplicated()
        .sum()
    )


    st.metric(
        "Duplicate Records",
        int(duplicate_records)
    )


    if duplicate_records > 0:

        st.dataframe(
            df[df.duplicated()],
            use_container_width=True
        )

    else:

        st.success(
            "No duplicate records found."
        )


# =========================================================
# DATA VALIDATION
# =========================================================

elif st.session_state.page == "Validation":

    st.header("✅ Data Validation")


    validation_df = df.copy()


    # =====================================================
    # NUMERIC VALIDATION
    # =====================================================

    numeric_columns = (
        validation_df
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )


    if len(numeric_columns) > 0:

        st.subheader(
            "🔢 Numerical Value Validation"
        )


        validation_results = []


        for col in numeric_columns:

            missing_count = (
                validation_df[col]
                .isnull()
                .sum()
            )


            negative_count = (
                validation_df[col]
                .lt(0)
                .sum()
            )


            validation_results.append({

                "Column":
                    col,

                "Missing Values":
                    int(missing_count),

                "Negative Values":
                    int(negative_count)

            })


        validation_report = pd.DataFrame(
            validation_results
        )


        st.dataframe(
            validation_report,
            use_container_width=True
        )


    else:

        st.warning(
            "No numeric columns detected."
        )


    # =====================================================
    # DATE VALIDATION
    # =====================================================

    st.subheader(
        "📅 Date Validation"
    )


    date_columns = [

        col

        for col in df.columns

        if "date" in col.lower()

    ]


    if len(date_columns) > 0:

        date_results = []


        for col in date_columns:

            converted_dates = pd.to_datetime(
                df[col],
                errors="coerce"
            )


            invalid_dates = (

                converted_dates.isnull()

                &

                df[col].notnull()

            ).sum()


            date_results.append({

                "Column":
                    col,

                "Invalid Dates":
                    int(invalid_dates)

            })


        date_report = pd.DataFrame(
            date_results
        )


        st.dataframe(
            date_report,
            use_container_width=True
        )


    else:

        st.info(
            "No date columns detected."
        )


# =========================================================
# ANOMALY DETECTION
# =========================================================

elif st.session_state.page == "Anomaly":

    st.header("🤖 Anomaly Detection")


    # =====================================================
    # GET NUMERIC COLUMNS
    # =====================================================

    numeric_columns = (
        df
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )


    if len(numeric_columns) == 0:

        st.error(
            "No numeric columns were detected."
        )

        st.stop()


    # =====================================================
    # DISPLAY NUMERIC COLUMNS
    # =====================================================

    st.success(
        f"{len(numeric_columns)} "
        f"numeric column(s) detected."
    )


    st.write(
        "Detected numeric columns:"
    )


    st.code(
        "\n".join(
            numeric_columns
        )
    )


    # =====================================================
    # SELECT COLUMN
    # =====================================================

    selected_column = st.selectbox(
        "Select Numeric Column",
        numeric_columns
    )


    # =====================================================
    # PREPARE DATA
    # =====================================================

    temp_df = df.copy()


    temp_df = temp_df[
        temp_df[selected_column]
        .notna()
    ].copy()


    # =====================================================
    # Z-SCORE
    # =====================================================

    st.subheader(
        "📊 Z-Score Detection"
    )


    if len(temp_df) < 2:

        st.warning(
            "Not enough numerical values for Z-Score detection."
        )

    else:

        values = temp_df[
            selected_column
        ].astype(float)


        # Check if all values are identical

        if values.nunique() <= 1:

            st.info(
                "Z-Score cannot identify anomalies because "
                "all values in this column are identical."
            )

        else:

            temp_df["Z_Score"] = zscore(
                values
            )


            zscore_anomalies = temp_df[
                temp_df["Z_Score"].abs() > 3
            ]


            st.write(
                f"Anomalies Detected: "
                f"{len(zscore_anomalies)}"
            )


            if len(zscore_anomalies) > 0:

                st.dataframe(
                    zscore_anomalies,
                    use_container_width=True
                )

            else:

                st.success(
                    "No Z-Score anomalies detected."
                )


    # =====================================================
    # ISOLATION FOREST
    # =====================================================

    st.subheader(
        "🌲 Isolation Forest Detection"
    )


    if len(temp_df) < 10:

        st.warning(
            "Not enough numerical values for "
            "Isolation Forest detection."
        )

    else:

        model = IsolationForest(
            contamination=0.02,
            random_state=42
        )


        temp_df["Anomaly"] = model.fit_predict(
            temp_df[
                [selected_column]
            ]
        )


        isolation_anomalies = temp_df[
            temp_df["Anomaly"] == -1
        ]


        st.write(
            f"Anomalies Detected: "
            f"{len(isolation_anomalies)}"
        )


        if len(isolation_anomalies) > 0:

            st.dataframe(
                isolation_anomalies,
                use_container_width=True
            )

        else:

            st.success(
                "No Isolation Forest anomalies detected."
            )


# =========================================================
# DATA STRUCTURE
# =========================================================

elif st.session_state.page == "Structure":

    st.header("📈 Data Structure")


    structure_df = pd.DataFrame({

        "Column Name":
            df.columns,

        "Data Type":
            df.dtypes.astype(str)

    })


    st.dataframe(
        structure_df,
        use_container_width=True
    )


    # -----------------------------------------------------
    # NUMERIC DETECTION DETAILS
    # -----------------------------------------------------

    st.subheader(
        "🔢 Automatic Data Type Detection"
    )


    detection_df = pd.DataFrame(
        numeric_detection_results
    )


    st.dataframe(
        detection_df,
        use_container_width=True
    )


# =========================================================
# DATA QUALITY REPORT
# =========================================================

elif st.session_state.page == "Report":

    st.header("📥 Data Quality Report")


    total_records = df.shape[0]

    total_columns = df.shape[1]

    missing_values = (
        df.isnull()
        .sum()
        .sum()
    )

    duplicate_records = (
        df.duplicated()
        .sum()
    )


    total_cells = (
        total_records
        * total_columns
    )


    if total_cells > 0:

        quality_score = (
            (
                total_cells
                - missing_values
            )
            / total_cells
        ) * 100

    else:

        quality_score = 0


    report = pd.DataFrame({

        "Metric": [

            "Total Records",

            "Total Columns",

            "Missing Values",

            "Duplicate Records",

            "Data Quality Score"

        ],

        "Value": [

            total_records,

            total_columns,

            int(missing_values),

            int(duplicate_records),

            f"{quality_score:.2f}%"

        ]

    })


    st.dataframe(
        report,
        use_container_width=True
    )


    csv = report.to_csv(
        index=False
    )


    st.download_button(

        label="📥 Download Report",

        data=csv,

        file_name="data_quality_report.csv",

        mime="text/csv"

    )

