# 🎬 Movie Recommendation Engine

**Intelligent content-based and collaborative filtering system for personalized movie suggestions**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-ff69b4)
![GitHub stars](https://img.shields.io/github/stars/somya245/movie_recommender)

## 🍿 Features
- **Hybrid Recommendation** combining:
  - Content-based filtering (TF-IDF + Cosine Similarity)
  - Collaborative filtering (KNN + Matrix Factorization)
- **Web Interface** with Streamlit dashboard
- **Pre-trained Models** on MovieLens 25M dataset
- **Cold Start Solution** for new users/movies

## 🚀 Quick Start
``bash
git clone https://github.com/somya245/movie_recommender.git
cd movie_recommender
pip install -r requirements.txt

# Launch web interface
streamlit run app.py


## 📊 Performance (MovieLens 25M)
Algorithm	RMSE	Precision@10	Coverage
SVD	0.82	0.43	92%
KNN	0.85	0.39	85%
Hybrid	0.79	0.47	97%
## 📂 Repository Structure
text
movie_recommender/
├── data/                  # Processed datasets
│   ├── movies.csv         # Movie metadata
│   └── ratings.pkl        # User ratings
├── models/                # Trained models
├── app.py                 # Streamlit interface
├── recommender.py         # Core logic
└── train.py               # Model training
