import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "CustomerID" in df.columns:
        df = df.dropna(subset=["CustomerID"]).copy()
        df["CustomerID"] = df["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True)
    if "StockCode" in df.columns:
        df["StockCode"] = df["StockCode"].astype(str)
    return df

def build_user_item_matrix(df: pd.DataFrame, value_col: str = "Quantity") -> pd.DataFrame:
    df = _normalize_ids(df)
    uim = df.pivot_table(
        index="CustomerID",
        columns="StockCode",
        values=value_col,
        aggfunc="sum",
        fill_value=0,
    )
    return uim

def compute_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    sim = cosine_similarity(user_item_matrix)
    return pd.DataFrame(sim, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_products(user_id, user_item_matrix, similarity_df, df_original, top_n: int = 10):
    user_id = str(user_id).replace(".0", "")

    if user_id not in user_item_matrix.index:
        return pd.DataFrame({"Message": [f"Customer {user_id} not found."]})

    # Extract similarity vector and ensure alignment with user_item_matrix.index
    sim_vec = similarity_df.loc[user_id]
    if not sim_vec.index.equals(user_item_matrix.index):
        sim_vec = sim_vec.reindex(user_item_matrix.index, fill_value=0)

    # Remove the user's own similarity score (set to 0 to avoid self-influence)
    if user_id in sim_vec.index:
        sim_vec.loc[user_id] = 0

    # Compute denominator (sum of similarities)
    denom = sim_vec.sum()
    if denom == 0:
        return pd.DataFrame({"Message": [f"No similar users found for {user_id}."]})

    # Compute weighted scores
    try:
        weighted_scores = user_item_matrix.T.dot(sim_vec) / denom
    except ValueError as e:
        return pd.DataFrame({"Message": [f"Matrix alignment error for {user_id}: {str(e)}"]})

    # Exclude items the user has already purchased
    user_row = user_item_matrix.loc[user_id]
    already = user_row[user_row > 0].index
    weighted_scores = weighted_scores.drop(index=already, errors="ignore")

    if weighted_scores.empty:
        return pd.DataFrame({"Message": [f"No unseen items to recommend for {user_id}."]})

    # Get product metadata
    meta = (
        _normalize_ids(df_original)
        .drop_duplicates(subset=["StockCode"])[["StockCode", "Description"]]
        .set_index("StockCode")
    )

    # Build recommendations DataFrame
    recs = (
        pd.DataFrame(
            {"StockCode": weighted_scores.index, "Estimated Score": weighted_scores.values}
        )
        .join(meta, on="StockCode")
        .sort_values("Estimated Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    if "Description" not in recs.columns:
        recs["Description"] = "N/A"

    return recs[["StockCode", "Description", "Estimated Score"]]    

def recommend_products22(user_id, user_item_matrix, similarity_df, df_original, top_n: int = 10):
    user_id = str(user_id).replace(".0", "")

    if user_id not in user_item_matrix.index:
        return pd.DataFrame({"Message": [f"Customer {user_id} not found."]})

    sim_vec = similarity_df.loc[user_id].reindex(user_item_matrix.index).fillna(0)

    if user_id in sim_vec.index:
        sim_vec = sim_vec.drop(user_id)

    denom = sim_vec.sum()
    if denom == 0:
        return pd.DataFrame({"Message": [f"No similar users found for {user_id}."]})

    weighted_scores = user_item_matrix.T.dot(sim_vec) / denom

    user_row = user_item_matrix.loc[user_id]
    already = user_row[user_row > 0].index
    weighted_scores = weighted_scores.drop(index=already, errors="ignore")

    if weighted_scores.empty:
        return pd.DataFrame({"Message": [f"No unseen items to recommend for {user_id}."]})

    meta = (
        _normalize_ids(df_original)
        .drop_duplicates(subset=["StockCode"])[["StockCode", "Description"]]
        .set_index("StockCode")
    )

    recs = (
        pd.DataFrame(
            {"StockCode": weighted_scores.index, "Estimated Score": weighted_scores.values}
        )
        .join(meta, on="StockCode")
        .sort_values("Estimated Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    if "Description" not in recs.columns:
        recs["Description"] = "N/A"

    return recs[["StockCode", "Description", "Estimated Score"]]

def load_and_prepare_data(data_path: str = "data/cleaned_data.csv", value_col: str = "Quantity"):
    df = pd.read_csv(data_path)
    df = _normalize_ids(df)
    uim = build_user_item_matrix(df, value_col=value_col)
    sim_df = compute_similarity(uim)
    return df, uim, sim_df
