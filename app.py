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
