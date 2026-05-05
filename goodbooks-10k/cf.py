import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. DATA PREPARATION
ratings = pd.read_csv("ratings.csv")
books = pd.read_csv("books.csv")

# Create mappings to translate IDs to matrix row/column indices
user_map = {id: i for i, id in enumerate(ratings['user_id'].unique())}
book_map = {id: i for i, id in enumerate(ratings['book_id'].unique())}
inv_book_map = {i: id for id, i in book_map.items()}

ratings['u_idx'] = ratings['user_id'].map(user_map)
ratings['b_idx'] = ratings['book_id'].map(book_map)

# 2. DATA SPLIT
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

# 3. CREATE SPARSE MATRIX
# We use a sparse matrix to save memory; it only stores the known ratings
train_matrix = csr_matrix(
    (train_df['rating'], (train_df['u_idx'], train_df['b_idx'])),
    shape=(len(user_map), len(book_map))
)

# 4. MODEL (TRUNCATED SVD)
# n_components (latent factors) represents the hidden "features" of books/users
svd = TruncatedSVD(n_components=50, random_state=42)
user_features = svd.fit_transform(train_matrix)
book_features = svd.components_

# 5. EVALUATION (RMSE)
# Predict ratings for the test set pairs
test_preds = [np.dot(user_features[row.u_idx], book_features[:, row.b_idx])
              for row in test_df.itertuples()]
rmse = np.sqrt(mean_squared_error(test_df['rating'], test_preds))
print(f"Test RMSE: {rmse:.4f}")

# 6. RECOMMENDATION FUNCTION
def get_recommendations(user_id, n=5):
    if user_id not in user_map: return "User ID not found."

    u_idx = user_map[user_id]
    # Reconstruct the user's row in the matrix (User Vector * Item Matrix)
    preds = np.dot(user_features[u_idx], book_features)

    # Filter out books the user has already read
    read_books = set(ratings[ratings['user_id'] == user_id]['b_idx'])

    # Sort predictions and fetch top N unread books
    sorted_indices = np.argsort(preds)[::-1]
    recommendations = []
    for idx in sorted_indices:
        if idx not in read_books:
            b_id = inv_book_map[idx]
            title = books[books['id'] == b_id]['title'].values[0]
            recommendations.append({"Title": title, "Score": round(preds[idx], 2)})
        if len(recommendations) == n: break

    return pd.DataFrame(recommendations)

# Example:
print(get_recommendations(user_id=314))