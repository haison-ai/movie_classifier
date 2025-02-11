import numpy as np
import pandas as pd


def load_user_rating_data(df: pd.DataFrame, n_users: int, n_movies: int):
    """
    Convert the user and movie ratings into matrix
    """
    data = np.zeros([n_users, n_movies], dtype=np.intc)
    movie_id_mapping = {}

    for user_id, movie_id, rating in zip(df["user_id"], df["movie_id"], df["rating"]):
        user_id = int(user_id) - 1

        if movie_id not in movie_id_mapping:
            movie_id_mapping[movie_id] = len(movie_id_mapping)

        data[user_id, movie_id_mapping[movie_id]] = rating

    return data, movie_id_mapping
