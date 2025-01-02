import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


ratings_dict = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    'movie_id': [1, 2, 3, 1, 2, 4, 2, 3, 4, 1, 4],
    'rating': [5, 3, 4, 4, 2, 5, 2, 4, 4, 3, 5]
}

ratings_df = pd.DataFrame(ratings_dict)

user_item_matrix = ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating')

user_item_matrix = user_item_matrix.fillna(0)

print("User-Item Matrix:")
print(user_item_matrix)

user_similarity = cosine_similarity(user_item_matrix)

user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

print("\nUser Similarity Matrix:")
print(user_similarity_df)

def get_recommendations(user_id, user_item_matrix, user_similarity_df, num_recommendations=5):
  
    user_sim_scores = user_similarity_df[user_id]
    weighted_ratings = user_sim_scores.values @ user_item_matrix.values
    weighted_ratings_series = pd.Series(weighted_ratings, index=user_item_matrix.columns)
    user_rated_movies = user_item_matrix.loc[user_id]
    weighted_ratings_series = weighted_ratings_series[user_rated_movies == 0]

    recommendations = weighted_ratings_series.nlargest(num_recommendations)
    return recommendations
user_id = 1
recommendations = get_recommendations(user_id, user_item_matrix, user_similarity_df)
print(f"\nTop recommendations for User {user_id}:")
print(recommendations)