"""
Advanced Movie Recommendation System - Training Pipeline
Adapted for Lite TMDB Dataset
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from nltk.stem.snowball import SnowballStemmer
import pickle
import json
from pathlib import Path
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')


class MovieRecommenderTrainer:
    def __init__(self, output_dir='./models', use_dimensionality_reduction=True, n_components=500):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_svd = use_dimensionality_reduction
        self.n_components = n_components
        self.stemmer = SnowballStemmer('english')
        
    def load_data(self, data_path):
        print("Loading TMDB dataset...")
        # Force it to read the file directly
        df = pd.read_csv(data_path, low_memory=False)
        print(f"Loaded {len(df)} movies")
        print(f"Columns found: {df.columns.tolist()}")
        return df
    
    def clean_and_engineer_features(self, df, quality_threshold='medium'):
        print("Engineering features...")
        
        # Filter by quality threshold
        thresholds = {'low': 5, 'medium': 50, 'high': 500}
        min_votes = thresholds.get(quality_threshold, 50)
        df = df[df['vote_count'] >= min_votes].copy()
        print(f"Filtered to {len(df)} movies with {min_votes}+ votes")
        
        # --- SILENCED MISSING COLUMNS ---
        # df = df[df['status'] == 'Released'].copy()
        # df['genres'] = df['genres'].apply(lambda x: self.parse_json_column(x, 'name'))
        # df['keywords'] = df['keywords'].apply(lambda x: self.parse_json_column(x, 'name'))
        
        # Process overview (plot summary)
        df['overview_clean'] = df['overview'].fillna('').astype(str)
        df['overview_words'] = df['overview_clean'].apply(
            lambda x: [word.lower() for word in x.split()[:50]]
        )
        
        # Create comprehensive soup feature (Simplified to only use Overview)
        print("Creating feature soup from overview...")
        df['soup'] = df['overview_words'].apply(lambda x: ' '.join(x) if x else '')
        
        # Filter valid entries
        df = df[df['soup'].str.len() > 10].copy()
        df = df.dropna(subset=['title'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # Sort by popularity
        df['quality_score'] = df['vote_average'] * np.log1p(df['vote_count'])
        df = df.sort_values('quality_score', ascending=False)
        
        # Ensure ID column is present
        if 'tconst' in df.columns and 'imdb_id' not in df.columns:
            df['imdb_id'] = df['tconst']
        elif 'id' in df.columns:
            df['imdb_id'] = df['id']

        df = df.reset_index(drop=True)
        print(f"Processed {len(df)} valid movies")
        return df
    
    def build_tfidf_matrix(self, df):
        print("Building TF-IDF matrix...")
        n_movies = len(df)
        max_features = 10000 if n_movies < 10000 else 15000
        
        tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english',
            max_features=max_features,
            sublinear_tf=True
        )
        
        tfidf_matrix = tfidf.fit_transform(df['soup'])
        return tfidf_matrix, tfidf
    
    def compute_similarity_matrix(self, tfidf_matrix):
        if self.use_svd and tfidf_matrix.shape[0] > 1000:
            print(f"Applying SVD reduction...")
            n_components = min(self.n_components, tfidf_matrix.shape[0] - 1)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_matrix = svd.fit_transform(tfidf_matrix)
            similarity_matrix = cosine_similarity(reduced_matrix)
            return similarity_matrix.astype(np.float32), svd
        else:
            print("Computing similarity...")
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            return similarity_matrix.astype(np.float32), None
    
    def save_model(self, df, similarity_matrix, tfidf_vectorizer, svd_model=None):
        print("Saving model artifacts...")
        
        # Only save columns that actually exist in your CSV
        available_cols = [c for c in ['id', 'title', 'release_date', 'vote_average', 'vote_count', 'popularity', 'overview', 'imdb_id'] if c in df.columns]
        metadata_df = df[available_cols].copy()
        
        metadata_df.to_parquet(self.output_dir / 'movie_metadata.parquet', compression='gzip', index=True)
        
        np.save(self.output_dir / 'similarity_matrix.npy', similarity_matrix)
        
        title_to_idx = pd.Series(df.index, index=df['title']).to_dict()
        with open(self.output_dir / 'title_to_idx.json', 'w') as f:
            json.dump(title_to_idx, f)
        
        with open(self.output_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        
        if svd_model:
            with open(self.output_dir / 'svd_model.pkl', 'wb') as f:
                pickle.dump(svd_model, f)
        
        print(f"✅ Model saved to {self.output_dir}")

    def train(self, data_path, quality_threshold='medium', max_movies=None):
        print("="*80)
        print("🎬 TMDB Movie Recommendation System Training")
        print("="*80)
        
        df = self.load_data(data_path)
        df = self.clean_and_engineer_features(df, quality_threshold)
        
        if max_movies and len(df) > max_movies:
            df = df.head(max_movies)
        
        tfidf_matrix, tfidf_vectorizer = self.build_tfidf_matrix(df)
        similarity_matrix, svd_model = self.compute_similarity_matrix(tfidf_matrix)
        self.save_model(df, similarity_matrix, tfidf_vectorizer, svd_model)
        
        return df, similarity_matrix

if __name__ == "__main__":
    # Ensure this matches your file name exactly
    path = "movies.csv" 
    
    trainer = MovieRecommenderTrainer(
        output_dir='./models',
        use_dimensionality_reduction=True,
        n_components=500
    )
    
    df, sim_matrix = trainer.train(
        path, 
        quality_threshold='medium', 
        max_movies=10000 
    )
    
    print(f"\n📊 Training Complete!")
    print(f"Movies processed: {len(df)}")