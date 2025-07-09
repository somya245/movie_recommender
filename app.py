import streamlit as st
import pandas as pd
import joblib

def main():
    st.set_page_config(
        page_title="🎬 Movie Recommender",
        page_icon="🍿",
        layout="wide"
    )
    
    @st.cache_data
    def load_data():
        try:
            movies = joblib.load("model/movies_metadata.pkl")
            similarity = joblib.load("model/similarity_matrix.pkl") 
            movie_ids = joblib.load("model/movie_ids.pkl")
            return movies, similarity, movie_ids
        except FileNotFoundError:
            st.error("Model files not found. Please run train.py first")
            st.stop()

    movies, similarity, movie_ids = load_data()
    
    st.title("Movie Recommender System")
    selected_movie = st.selectbox(
        "Select a movie you like:",
        movies['title'].values
    )

    if st.button("Get Recommendations"):
        # Find the index of the selected movie
        idx = movies[movies['title'] == selected_movie].index[0]
        # Get similarity scores for this movie
        sim_scores = list(enumerate(similarity[idx]))
        # Sort movies by similarity score (excluding the selected movie itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        # Get the indices of the recommended movies
        recommended_indices = [i[0] for i in sim_scores]
        # Display recommended movie titles
        st.subheader("Recommended Movies:")
        for i in recommended_indices:
            st.write(movies.iloc[i]['title'])

if __name__ == "__main__":
    main()  # Properly encapsulated Streamlit app