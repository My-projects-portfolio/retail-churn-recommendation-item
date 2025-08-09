import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd

from src.churn_risk_classifier import prepare_churn_features
from src.collaborative_model import load_and_prepare_data, recommend_products
from src.popularity_model import get_popular_items

@st.cache_data(show_spinner=False)
def load_customer_risk_data(path: str = "data/cleaned_data.csv") -> pd.DataFrame:
    df = prepare_churn_features(path)
    required = {"CustomerID", "risk_level", "days_since_last_purchase", "frequency", "monetary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Churn features missing columns: {missing}")
    return df

@st.cache_resource(show_spinner=False)
def load_model_resources(path: str = "data/cleaned_data.csv"):
    df_model, user_item_matrix, similarity_df = load_and_prepare_data(path)

    # normalize IDs (belt & braces)
    fix = lambda s: s.astype(str).str.replace(r"\.0$", "", regex=True)
    user_item_matrix.index = fix(user_item_matrix.index)
    user_item_matrix.columns = fix(user_item_matrix.columns)
    similarity_df.index = fix(similarity_df.index)
    similarity_df.columns = fix(similarity_df.columns)

    # sanity for dot-product
    if not user_item_matrix.index.equals(similarity_df.index) or not user_item_matrix.index.equals(similarity_df.columns):
        raise ValueError("User-item and similarity indices/columns are not aligned. Check ID normalization.")
    return df_model, user_item_matrix, similarity_df

def parse_customer_id(raw: str) -> str | None:
    if not raw or not raw.strip():
        return None
    try:
        return str(int(float(raw.strip())))
    except ValueError:
        return None

def main():
    st.set_page_config(page_title="Churn + Recommendations", page_icon="🧠", layout="wide")
    st.title("🧠 Customer Churn Risk + Product Recommendation System")

    with st.sidebar:
        st.header("⚙️ Settings")
        data_path = st.text_input("Data path", value="data/cleaned_data.csv")
        st.caption("Needs columns: CustomerID, StockCode, Quantity, Description, UnitPrice, InvoiceNo, InvoiceDate.")

        st.markdown("---")
        st.subheader("📈 Popular Items (fallback)")
        top_n = st.slider("Top-N", 5, 30, 10)
        rank_by = st.radio("Rank by", ["TotalPrice", "Quantity"], horizontal=True)

        if st.button("Preview popular items"):
            try:
                popular_df = get_popular_items(data_path, top_n=top_n, by=rank_by)
                st.dataframe(popular_df, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load popular items: {e}")

    try:
        with st.spinner("Loading churn features..."):
            churn_df = load_customer_risk_data(data_path)

        with st.spinner("Loading recommender resources..."):
            df_model, user_item_matrix, similarity_df = load_model_resources(data_path)

    except Exception as e:
        st.error("❌ Failed to load data/resources.")
        st.exception(e)
        st.stop()

    # ---- Churn Explorer ----
    st.subheader("🔍 Churn Explorer")
    c1, c2 = st.columns([1, 2])
    with c1:
        risk_level = st.selectbox("Risk Level", ["High Risk", "Medium Risk", "Low Risk"])
        filtered = churn_df[churn_df["risk_level"] == risk_level].copy()
        st.markdown(f"**{len(filtered)} customers** in {risk_level}")
    with c2:
        st.caption("Sorted by monetary (desc).")
        st.dataframe(
            filtered[["CustomerID", "days_since_last_purchase", "frequency", "monetary"]]
            .sort_values("monetary", ascending=False),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # ---- Recommendations ----
    st.subheader("🎯 Get Recommendations by CustomerID")
    st.caption("Tip: Enter numeric CustomerID (e.g., 12350). If you see 12350.0 in tables, just type 12350.")
    ic1, ic2 = st.columns([2, 1])
    with ic1:
        cid_raw = st.text_input("CustomerID", value="", placeholder="e.g., 12350")
    with ic2:
        top_k = st.slider("Top-N", 5, 30, 10)

    if st.button("Get Recommendations"):
        cid = parse_customer_id(cid_raw)
        if cid is None:
            st.error("CustomerID must be numeric (e.g., 12350).")
            st.stop()

        if cid not in user_item_matrix.index:
            st.error(f"CustomerID {cid} not found in the dataset.")
            st.info("Try the Popular Items preview in the sidebar as a fallback.")
            st.stop()

        with st.spinner("Generating personalized recommendations..."):
            try:
                recs = recommend_products(cid, user_item_matrix, similarity_df, df_model, top_n=top_k)
            except Exception as e:
                st.error("❌ Failed to generate recommendations.")
                st.exception(e)
                st.stop()

        if recs is None or recs.empty:
            st.info(f"No recommendations available for Customer {cid}.")
        elif "Message" in recs.columns:
            st.info(recs["Message"].iloc[0])
        else:
            st.success(f"Top {len(recs)} Product Recommendations for Customer {cid}")
            show_cols = [c for c in ["StockCode", "Description", "Estimated Score"] if c in recs.columns]
            st.dataframe(recs[show_cols], use_container_width=True, hide_index=True)

    # ---- Data Health ----
    with st.expander("🩺 Data health checks"):
        st.write("**User-item matrix shape**:", user_item_matrix.shape)
        st.write("**Similarity matrix shape**:", similarity_df.shape)
        try:
            assert user_item_matrix.index.equals(similarity_df.index)
            assert user_item_matrix.index.equals(similarity_df.columns)
            st.success("Indices aligned for dot-products ✅")
        except AssertionError:
            st.error("Indices are NOT aligned. Check ID normalization in collaborative_model.py.")

if __name__ == "__main__":
    main()
