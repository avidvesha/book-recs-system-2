import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load and Prepare the Data
df = pd.read_csv('cbf_final.csv', index_col=0)
display(df)

# Clean the tags: replace commas with spaces so it's a clean string
df['tag_name_clean'] = df['tag_name'].str.replace(',', ' ')

# 2. Create the "Bag of Words" (Content Soup)
# We combine authors, decades, and tags into one training string
df['soup'] = df['authors'] + " " + df['decade_year'] + " " + df['tag_name_clean']

tokenized_soup = df['soup'].apply(lambda x: x.split())

# 3. Vectorizing using Word2Vec
# training a model to create 100-dimensional vectors for each token
w2v_model = Word2Vec(sentences=tokenized_soup, vector_size=100, window=5, min_count=1, workers=4, epochs=10)

# Function to generate a single vector for a book by averaging its word vectors
def get_book_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Create a matrix of all book vectors
book_vectors = np.array([get_book_vector(t, w2v_model) for t in tokenized_soup])

# 4. Calculate Cosine Similarity
# This creates a map of how similar every book is to every other book
cosine_sim = cosine_similarity(book_vectors)

# 5. Final Recommendation Function
def recommend_books(title, cosine_sim=cosine_sim, df=df):
    try:
        # Find the index of the book that matches the title
        idx = df[df['title'] == title].index[0]

        # Get the pairwise similarity scores of all books with that book
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the books based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        print(sim_scores)

        # Get the scores of the 5 most similar books (skipping the first one)
        sim_scores = sim_scores[1:11]

        # Get the book indices
        book_indices = [i[0] for i in sim_scores]

        # Return the top 5 most similar books
        return df[['title', 'authors']].iloc[book_indices]
    except IndexError:
        return "Book title not found in dataset."

# Example Usage:
print(recommend_books("Mockingjay (The Hunger Games, #3)"))