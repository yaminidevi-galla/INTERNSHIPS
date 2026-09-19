
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(
    page_title="Intelligent Product Recommendation",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Intelligent Product Recommendation & Customer Purchase Pattern Analysis")
st.caption("Clustering & Association — Apriori / Market Basket Analysis")

# -----------------------------
# Sample data from the case study
# -----------------------------
SAMPLE_DATA = pd.DataFrame({
    "TransactionID": ["O001", "O002", "O003", "O004", "O005", "O006", "O007"],
    "ProductName": [
        "Smartphone, Mobile Case",
        "Smartphone, Screen Protector",
        "Smartphone, Mobile Case, Earphones",
        "Laptop, Wireless Mouse",
        "Laptop, Laptop Bag, Wireless Mouse",
        "Tablet, Stylus",
        "Smartphone, Earphones"
    ]
})

def make_sample_rows():
    rows = []
    for _, row in SAMPLE_DATA.iterrows():
        for product in [p.strip() for p in row["ProductName"].split(",")]:
            rows.append({
                "TransactionID": row["TransactionID"],
                "ProductName": product,
                "Quantity": 1,
                "Date": "",
                "CustomerID": "",
                "UnitPrice": "",
                "Category": ""
            })
    return pd.DataFrame(rows)

def normalize_columns(df):
    """Map common column names to TransactionID and ProductName."""
    df = df.copy()
    lookup = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}

    transaction_aliases = [
        "transactionid", "transaction", "orderid", "order", "invoice", "invoiceno"
    ]
    product_aliases = [
        "productname", "product", "item", "itemname", "description"
    ]

    transaction_col = next((lookup[x] for x in transaction_aliases if x in lookup), None)
    product_col = next((lookup[x] for x in product_aliases if x in lookup), None)

    if transaction_col is None or product_col is None:
        raise ValueError(
            "CSV must contain a transaction column and a product column. "
            "Examples: TransactionID + ProductName, or OrderID + Product."
        )

    df = df.rename(columns={
        transaction_col: "TransactionID",
        product_col: "ProductName"
    })

    df["TransactionID"] = df["TransactionID"].astype(str).str.strip()
    df["ProductName"] = df["ProductName"].astype(str).str.strip()

    return df

def preprocess_data(df):
    df = normalize_columns(df)

    # Remove missing transaction/product values
    df = df[
        df["TransactionID"].notna()
        & df["ProductName"].notna()
        & (df["TransactionID"].str.lower() != "nan")
        & (df["ProductName"].str.lower() != "nan")
        & (df["TransactionID"] != "")
        & (df["ProductName"] != "")
    ].copy()

    # Optional quantity validation
    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df = df[df["Quantity"].isna() | (df["Quantity"] > 0)].copy()

    # Remove duplicate transaction-product records
    df = df.drop_duplicates(subset=["TransactionID", "ProductName"])

    # Clean product names
    df["ProductName"] = (
        df["ProductName"]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    return df

def create_basket(df):
    basket = (
        df.groupby(["TransactionID", "ProductName"])
        .size()
        .unstack(fill_value=0)
    )

    # Binary encoding: 1 = purchased, 0 = not purchased
    basket = (basket > 0).astype(int)
    return basket

def format_itemset(x):
    if isinstance(x, frozenset):
        return ", ".join(sorted(map(str, x)))
    return str(x)

def generate_rules(basket, min_support, min_confidence, min_lift):
    frequent = apriori(
        basket,
        min_support=min_support,
        use_colnames=True
    )

    if frequent.empty:
        return frequent, pd.DataFrame()

    try:
        rules = association_rules(
            frequent,
            metric="confidence",
            min_threshold=min_confidence
        )
    except TypeError:
        # Compatibility with mlxtend versions that require num_itemsets
        rules = association_rules(
            frequent,
            num_itemsets=len(basket),
            metric="confidence",
            min_threshold=min_confidence
        )

    if not rules.empty:
        rules = rules[rules["lift"] >= min_lift].copy()
        rules["antecedents_text"] = rules["antecedents"].apply(format_itemset)
        rules["consequents_text"] = rules["consequents"].apply(format_itemset)
        rules = rules.sort_values(
            ["confidence", "lift", "support"],
            ascending=False
        ).reset_index(drop=True)

    return frequent, rules

def recommend(product, rules, top_n=5):
    if rules.empty:
        return pd.DataFrame()

    product = product.strip().lower()
    matching = []

    for _, row in rules.iterrows():
        antecedents = {str(x).strip().lower() for x in row["antecedents"]}
        if product in antecedents:
            for item in row["consequents"]:
                matching.append({
                    "Recommended Product": item,
                    "Support": row["support"],
                    "Confidence": row["confidence"],
                    "Lift": row["lift"],
                    "Rule": f"{row['antecedents_text']} → {row['consequents_text']}"
                })

    if not matching:
        return pd.DataFrame()

    result = pd.DataFrame(matching)

    # Keep the strongest rule for each recommended product
    result = (
        result.sort_values(
            ["Confidence", "Lift", "Support"],
            ascending=False
        )
        .drop_duplicates(subset=["Recommended Product"])
        .head(top_n)
        .reset_index(drop=True)
    )

    return result

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Settings")

data_source = st.sidebar.radio(
    "Choose data source",
    ["Case-study sample data", "Upload CSV"]
)

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )
    if uploaded is None:
        st.info("Upload a CSV file to begin, or select 'Case-study sample data'.")
        st.stop()
    raw_df = pd.read_csv(uploaded)
else:
    raw_df = make_sample_rows()

st.sidebar.subheader("Apriori thresholds")
min_support = st.sidebar.slider(
    "Minimum Support",
    min_value=0.01,
    max_value=1.00,
    value=0.05,
    step=0.01
)
min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.01,
    max_value=1.00,
    value=0.50,
    step=0.01
)
min_lift = st.sidebar.number_input(
    "Minimum Lift",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

top_n = st.sidebar.number_input(
    "Number of recommendations",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

# -----------------------------
# Preprocessing
# -----------------------------
try:
    clean_df = preprocess_data(raw_df)
except ValueError as e:
    st.error(str(e))
    st.stop()

if clean_df.empty:
    st.error("No valid transaction records remain after preprocessing.")
    st.stop()

basket = create_basket(clean_df)
frequent_itemsets, rules = generate_rules(
    basket,
    min_support,
    min_confidence,
    min_lift
)

# -----------------------------
# Dashboard
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🧹 Preprocessing",
    "🛍️ Purchase Patterns",
    "🔗 Association Rules",
    "💡 Recommendation Engine"
])

with tab1:
    st.subheader("Project Overview")
    st.write(
        "This system analyzes historical shopping transactions, discovers "
        "frequent product combinations using Apriori, generates association "
        "rules using support, confidence and lift, and recommends products "
        "related to a customer-selected item."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", basket.shape[0])
    c2.metric("Products", basket.shape[1])
    c3.metric("Frequent Itemsets", len(frequent_itemsets))
    c4.metric("Strong Rules", len(rules))

    st.subheader("Transaction Data")
    st.dataframe(clean_df, use_container_width=True)

with tab2:
    st.subheader("Data Preprocessing")
    st.write("The preprocessing stage performs the operations described in the case study:")
    st.markdown("""
    - Removes rows without transaction or product information.
    - Removes duplicate transaction-product records.
    - Examines/removes invalid non-positive quantities when Quantity is available.
    - Cleans product names.
    - Groups products by transaction.
    - Converts transactions into binary 0/1 format.
    """)

    st.write("### Cleaned transaction records")
    st.dataframe(clean_df, use_container_width=True)

    st.write("### Binary transaction matrix")
    st.dataframe(basket, use_container_width=True)

    csv_bytes = basket.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download binary transaction matrix",
        data=csv_bytes,
        file_name="transaction_matrix.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("Purchase Pattern Analysis")

    product_frequency = basket.sum().sort_values(ascending=False)

    st.write("### Product frequency")
    freq_df = product_frequency.reset_index()
    freq_df.columns = ["Product", "Transactions"]
    st.dataframe(freq_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    product_frequency.plot(kind="bar", ax=ax)
    ax.set_title("Product Purchase Frequency")
    ax.set_xlabel("Product")
    ax.set_ylabel("Number of Transactions")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write("### Frequent itemsets")
    if frequent_itemsets.empty:
        st.warning("No frequent itemsets found. Lower the minimum support.")
    else:
        fi = frequent_itemsets.copy()
        fi["Itemsets"] = fi["itemsets"].apply(format_itemset)
        fi["Support"] = fi["support"].round(4)
        st.dataframe(
            fi[["Itemsets", "Support"]],
            use_container_width=True
        )

with tab4:
    st.subheader("Market Basket Analysis & Association Rules")

    st.write(
        "Association rules are represented as Antecedent → Consequent. "
        "Rules are filtered using the selected confidence and lift thresholds."
    )

    if rules.empty:
        st.warning(
            "No strong association rules were found. "
            "Try lowering Minimum Support, Minimum Confidence, or Minimum Lift."
        )
    else:
        display_rules = rules[
            [
                "antecedents_text",
                "consequents_text",
                "support",
                "confidence",
                "lift"
            ]
        ].copy()

        display_rules.columns = [
            "Antecedent",
            "Consequent",
            "Support",
            "Confidence",
            "Lift"
        ]

        display_rules["Support"] = display_rules["Support"].round(4)
        display_rules["Confidence"] = display_rules["Confidence"].round(4)
        display_rules["Lift"] = display_rules["Lift"].round(4)

        st.dataframe(display_rules, use_container_width=True)

        st.write("### Confidence of top rules")
        chart_df = display_rules.head(10).copy()
        chart_df["Rule"] = (
            chart_df["Antecedent"] + " → " + chart_df["Consequent"]
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(chart_df["Rule"], chart_df["Confidence"])
        ax.set_title("Association Rule Confidence")
        ax.set_xlabel("Rule")
        ax.set_ylabel("Confidence")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        rules_csv = display_rules.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download association rules",
            data=rules_csv,
            file_name="association_rules.csv",
            mime="text/csv"
        )

with tab5:
    st.subheader("💡 Recommendation Engine")

    products = sorted(basket.columns.tolist())

    if not products:
        st.warning("No products available.")
    else:
        selected_product = st.selectbox(
            "Select a product",
            products
        )

        recommendations = recommend(
            selected_product,
            rules,
            top_n=int(top_n)
        )

        if recommendations.empty:
            st.warning(
                f"No recommendation rule was found for '{selected_product}'. "
                "Try another product or lower the thresholds."
            )
        else:
            st.success(f"Recommendations for: {selected_product}")

            for i, row in recommendations.iterrows():
                st.markdown(
                    f"### {i + 1}. {row['Recommended Product']}"
                )
                st.write(
                    f"**Confidence:** {row['Confidence']:.2%}  |  "
                    f"**Support:** {row['Support']:.2%}  |  "
                    f"**Lift:** {row['Lift']:.2f}"
                )
                st.caption(f"Rule: {row['Rule']}")

            st.write("### Recommendation table")
            rec_display = recommendations.copy()
            rec_display["Support"] = rec_display["Support"].map(lambda x: f"{x:.2%}")
            rec_display["Confidence"] = rec_display["Confidence"].map(lambda x: f"{x:.2%}")
            rec_display["Lift"] = rec_display["Lift"].map(lambda x: f"{x:.2f}")
            st.dataframe(rec_display, use_container_width=True)

st.divider()
st.caption(
    "Academic implementation based on the uploaded IBM case study: "
    "Intelligent Product Recommendation & Customer Purchase Pattern Analysis."
)
