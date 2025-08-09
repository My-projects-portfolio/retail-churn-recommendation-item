Retail Churn + Recommendation System
A practical customer intelligence app that predicts churn risk and recommends top‑N products for each customer using real retail transactions.

Live Demo:
https://retail-churn-recommendation-item-abevbz4jtwheibxtyma7gk.streamlit.app/

🌟 What this app does
Churn Explorer
Classifies each customer into High / Medium / Low churn risk based on latency (days since last purchase) and monetary value compared to the population average.

Personalized Recommendations
Enter a CustomerID and get Top‑10 product recommendations using collaborative filtering (cosine similarity) over a user–item matrix.

One screen to act
Filter customers by risk → pick a customer (or type CustomerID) → get personalized items to re‑engage.

🧠 Dataset
Online Retail (UCI Machine Learning Repository)
Transactional data for a UK-based online retailer (2010–2011).
Typical columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.

Note: We use a cleaned file: data/cleaned_data.csv generated in earlier steps (removing cancellations, negative qty, missing IDs, and adding TotalPrice = Quantity * UnitPrice).

🎯 Project Goals
Identify churn risk for each customer based on recent activity and spend.

Recommend relevant products to re‑engage at‑risk customers (Top‑10 per customer).

Provide a simple Streamlit UI to filter by risk and request recommendations.

🧩 ML Models & Techniques
1) Churn Risk Classification (Heuristic)
Features (customer‑level):

days_since_last_purchase (recency / latency)

frequency (# of invoices)

monetary (sum of TotalPrice)

Risk rule (example):

High Risk: latency > avg latency and monetary < avg monetary

Medium Risk: latency > avg latency

Low Risk: otherwise

File: src/churn_risk_classifier.py → prepare_churn_features()

2) Collaborative Filtering (Cosine Similarity)
Build a user–item matrix: rows = CustomerID, cols = StockCode, values = Quantity (or TotalPrice).

Compute user‑to‑user cosine similarity.

For a target user, score items based on weighted purchases of similar users; exclude items already purchased.

Return Top‑10 with StockCode, Description, and an Estimated Score.

File: src/collaborative_model.py

load_and_prepare_data() → returns (df_original, user_item_matrix, similarity_df)

recommend_products(user_id, user_item_matrix, similarity_df, df_original, top_n=10)

Why cosine?
It avoids heavy native builds (no scikit-surprise) and runs smoothly on Streamlit Cloud.

3) (Optional) Content‑Based (TF‑IDF on Product Descriptions)
Vectorize Description with TF‑IDF and recommend textually similar items.

Useful when user history is sparse (cold start).

File: src/content_based_model.py (optional)

🖥️ App: Streamlit UI
Main file: app/streamlit_app.py

Features:

Churn Explorer: select risk level → see table of customers with latency, frequency, monetary.

Customer Input: type a CustomerID (e.g., 12350) → get Top‑10 recommendations.

📦 Project Structure
bash
Copy
Edit
retail-churn-recommendation/
├── app/
│   └── streamlit_app.py           # Streamlit UI
├── data/
│   └── cleaned_data.csv           # Cleaned dataset (required)
├── notebooks/                     # EDA / modeling (optional)
├── src/
│   ├── churn_risk_classifier.py   # Churn features + risk rules
│   ├── collaborative_model.py     # Cosine-based CF
│   ├── content_based_model.py     # (optional) TF-IDF content-based
│   └── popularity_model.py        # (optional) popularity baseline
├── requirements.txt
└── README.md
🚀 Run Locally
1) Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Minimal requirements.txt:

nginx
Copy
Edit
streamlit
pandas
scikit-learn
(If you added plotting or TF‑IDF, include matplotlib, seaborn, and scikit-learn is already there.)

2) Launch the app
bash
Copy
Edit
streamlit run app/streamlit_app.py
Open the URL shown (usually http://localhost:8501).

☁️ Run on Streamlit Cloud
Push this repo to GitHub.

Create a new Streamlit app pointing to app/streamlit_app.py.

Ensure requirements.txt is present.

Make sure data/cleaned_data.csv is in the repo (or load from a secure store).

Live Demo:
https://retail-churn-recommendation-item-abevbz4jtwheibxtyma7gk.streamlit.app/

🧪 Example Outputs
Churn Explorer (table):

python-repl
Copy
Edit
CustomerID | days_since_last_purchase | frequency | monetary | risk_level
---------- | ------------------------ | --------- | -------- | ----------
12350      | 97                       | 6         | 540.25   | High Risk
17841      | 12                       | 21        | 2980.10  | Low Risk
...
Top‑10 Recommendations (for Customer 12350):

python-repl
Copy
Edit
StockCode | Description                         | Estimated Score
--------- | ----------------------------------- | ---------------
85123A    | WHITE HANGING HEART T-LIGHT HOLDER  | 0.84
71053     | WHITE METAL LANTERN                 | 0.80
...
⚠️ Notes & Limitations
Heuristic churn: A simple rule-based approach (not a supervised classifier). You can upgrade to a labeled churn model if you define a churn window and train (e.g., Logistic Regression/XGBoost).

Cosine CF: Works well for known users with purchase history. New users/products (cold start) need popularity or content-based fallbacks.

Data quality: Ensure cleaned_data.csv has valid CustomerID, StockCode, Description, Quantity, UnitPrice, InvoiceDate, and TotalPrice.

🔮 Roadmap / Ideas
Add a hybrid ranker (blend collaborative + content-based + popularity).

Train a supervised churn model with a fixed cutoff (e.g., no purchase in last 90 days).

Add downloadable reports for high-risk customers and recommended campaigns.

Integrate with email/CRM for one-click re‑engagement.

🙏 Acknowledgements
Dataset: UCI Machine Learning Repository – Online Retail

Libraries: Streamlit, pandas, scikit‑learn

If you want, I can also add repository badges, a GIF screenshot, or a Makefile/Dockerfile for quick deployment.
