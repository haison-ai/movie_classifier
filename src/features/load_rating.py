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


def filter_target_movie(data, movie_id_mapping, target_movie_id):

    if target_movie_id not in movie_id_mapping:
        raise ValueError(f"Error: target_movie_id {target_movie_id} no está en movie_id_mapping")

    index = movie_id_mapping[target_movie_id]

    if index >= data.shape[1]:  # Verificar si el índice está fuera de los límites
        raise IndexError(f"Índice {index} fuera de los límites para data con {data.shape[1]} columnas")

    X_raw = np.delete(data, index, axis=1)
    Y_raw = data[:, index]

    X = X_raw[Y_raw > 0]
    Y = Y_raw[Y_raw > 0]

    return X, Y

def get_rating_counts(data):


    values, counts = np.unique(data, return_counts=True)
    return dict(zip(values, counts))

def count_samples(Y, recommend_threshold=3):

    Y[Y <= recommend_threshold] = 0
    Y[Y > recommend_threshold] = 1
    n_pos = (Y == 1).sum()
    n_neg = (Y == 0).sum()
    return n_pos, n_neg