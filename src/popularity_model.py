import pandas as pd

def get_popular_items(data_path, top_n=10, by='TotalPrice'):
    df = pd.read_csv(data_path)

    if 'TotalPrice' not in df.columns:
        if {'Quantity','UnitPrice'}.issubset(df.columns):
            df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        else:
            raise ValueError("TotalPrice not found and cannot be computed (need Quantity & UnitPrice).")

    if by not in ['Quantity', 'TotalPrice']:
        raise ValueError("Parameter 'by' must be either 'Quantity' or 'TotalPrice'")

    df['StockCode'] = df['StockCode'].astype(str)
    top_items = (
        df.groupby(['StockCode','Description'], dropna=False)[by]
          .sum()
          .sort_values(ascending=False)
          .head(top_n)
          .reset_index()
          .rename(columns={by: 'Score'})
    )
    return top_items
