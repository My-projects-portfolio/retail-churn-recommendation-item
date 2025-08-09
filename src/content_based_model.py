"""
Step 5: Content-Based Recommendation System
Recommends similar products based on product description using TF-IDF and cosine similarity.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
    def __init__(self, data_path):
        # Load dataset
        self.df = pd.read_csv(data_path)

        # Drop rows with missing descriptions
        self.df = self.df.dropna(subset=['Description'])

        # Normalize descriptions
        self.df['Description'] = self.df['Description'].str.lower()

        # Create TF-IDF matrix
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['Description'])

        # Compute cosine similarity matrix
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def get_similar_products(self, product_name, top_n=5):
        """
        Recommend top N similar products based on description similarity.
        
        Args:
            product_name (str): Name of the product (case-insensitive)
            top_n (int): Number of similar items to return
        
        Returns:
            DataFrame: Top N similar product descriptions and similarity scores
        """
        # Find index of the first matching product
        matches = self.df[self.df['Description'].str.contains(product_name.lower(), na=False)]
        if matches.empty:
            return f"No match found for '{product_name}'. Please try a different term."

        idx = matches.index[0]

        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

        # Fetch top N similar products
        similar_indices = [i[0] for i in sim_scores]
        similar_items = self.df.iloc[similar_indices][['StockCode', 'Description']].copy()
        similar_items['Similarity'] = [i[1] for i in sim_scores]

        return similar_items.reset_index(drop=True)


# Example usage
if __name__ == "__main__":
    recommender = ContentBasedRecommender("../data/cleaned_data.csv")
    product = "mug"
    print(f"Top similar products to '{product}':")
    print(recommender.get_similar_products(product, top_n=5))
