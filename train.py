import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import joblib
from tqdm import tqdm

def load_data():
    """Load and preprocess data"""
    print("Loading data...")
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

def create_sparse_matrix(ratings):
    """Create user-item sparse matrix"""
    print("Creating sparse matrix...")
    user_item_matrix = ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)
    return csr_matrix(user_item_matrix.values), user_item_matrix.columns

def compute_similarity(matrix):
    """Calculate cosine similarity"""
    print("Computing similarities (this may take several minutes)...")
    similarity = cosine_similarity(matrix.T)
    return similarity

def save_artifacts(similarity, movie_ids, movies):
    """Save model artifacts"""
    print("Saving artifacts...")
    joblib.dump(similarity, "model/similarity_matrix.pkl")
    joblib.dump(movie_ids, "model/movie_ids.pkl")
    joblib.dump(movies.set_index('movieId'), "model/movies_metadata.pkl")

if __name__ == "__main__":
    movies, ratings = load_data()
    sparse_matrix, movie_ids = create_sparse_matrix(ratings)
    similarity_matrix = compute_similarity(sparse_matrix)
    save_artifacts(similarity_matrix, movie_ids, movies)
    print("Training complete! Similarity matrix saved to model/")