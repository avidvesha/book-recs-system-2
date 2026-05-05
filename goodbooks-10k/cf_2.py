import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# 1. Load Datasets
ratings = pd.read_csv('ratings.csv')
books = pd.read_csv('books_clean.csv')

# 2. Filter Ratings: Only keep book_ids present in books_clean['id']
# Based on your instruction: ratings.book_id -> books_clean.id
valid_ids = books['id'].unique()
ratings_filtered = ratings[ratings['book_id'].isin(valid_ids)].copy()

# 3. Prepare data for Surprise
# The Reader defines the rating scale (usually 1-5 for books)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_filtered[['user_id', 'book_id', 'rating']], reader)

# 4. Data Split (80% Train, 20% Test)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 5. Model Training (SVD)
# We'll use standard SVD parameters; n_factors is the size of the latent space
model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
model.fit(trainset)

# 6. Evaluation
predictions = model.test(testset)
print(f"RMSE: {accuracy.rmse(predictions):.4f}")
print(f"MAE:  {accuracy.mae(predictions):.4f}")

# 7. Recommendation Function
def get_top_n_recommendations(user_id, n=5):
    # Get all book IDs the user hasn't rated yet
    rated_books = ratings_filtered[ratings_filtered['user_id'] == user_id]['book_id'].values
    all_books = books['id'].values
    books_to_predict = [b for b in all_books if b not in rated_books]
    
    # Predict ratings for unrated books
    preds = [model.predict(user_id, b_id) for b_id in books_to_predict]
    
    # Sort by estimated rating (est) and take top N
    preds.sort(key=lambda x: x.est, reverse=True)
    top_preds = preds[:n]
    
    # Map back to titles using books_clean
    rec_list = []
    for p in top_preds:
        title = books[books['id'] == p.iid]['title'].values[0]
        rec_list.append({"Book ID": p.iid, "Title": title, "Estimated Rating": round(p.est, 2)})
        
    return pd.DataFrame(rec_list)

# Example Usage:
# print(get_top_n_recommendations(user_id=12874, n=5))