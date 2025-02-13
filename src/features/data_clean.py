import logging
import os
from io import BytesIO

import boto3
import numpy as np
import pandas as pd

from src.features.load_rating import (count_ssamples, filter_target_movie,
                                      get_rating_counts, load_uer_rating_data)


class DataCleans3:
    def __init__(self, s3_name="movieclassifiers3", region="us-east-2"):
        """Inicia la conexión con AWS S3"""
        self.s3_name = s3_name
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)

    def read_s3_file(self, s3_route) -> pd.DataFrame:
        """Lee el archivo de ratings desde S3 y lo carga en un DataFrame"""
        try:
            csv = self.s3.get_object(Bucket=self.s3_name, Key=s3_route)
            df = pd.read_csv(
                BytesIO(csv["Body"].read()), delimiter="::", engine="python"
            )
            logging.info("Archivo cargado correctamente desde S3")
            return df
        except Exception as e:
            logging.error(f"Error al leer archivo desde S3: {e}")
            return None

    def transform_data(self, df: pd.DataFrame):
        df.columns = ["user_id", "movie_id", "rating", "timestamp"]
        print(df.head())

        n_users = df["user_id"].nunique()
        n_movies = df["movie_id"].nunique()
        print(f"Número de usuarios: {n_users}, Número de películas: {n_movies}")

        data, movie_id_mapping = load_user_rating_data(df, n_users, n_movies)

        values, counts = np.unique(data, return_counts=True)

        print("Distribución de ratings:")
        for value, count in zip(values, counts):
            print(f"Number of rating {value}: {count}")

        target_movie_id = df["movie_id"].value_counts().idxmax()
        print(f"Película con más ratings: {target_movie_id}")

        if target_movie_id not in movie_id_mapping:
            raise ValueError(
                f"Error: La película {target_movie_id} no está en movie_id_mapping"
            )

        X, Y = filter_target_movie(data, movie_id_mapping, target_movie_id)

        recommend_threshold = 3  # Ratings > 3 se consideran positivas
        Y_binary = np.copy(Y)  # Evitar modificar Y original

        Y_binary[Y_binary <= recommend_threshold] = 0  # No recomendado
        Y_binary[Y_binary > recommend_threshold] = 1  # Recomendado

        n_pos = (Y_binary == 1).sum()
        n_neg = (Y_binary == 0).sum()

        print(f"{n_pos} muestras positivas y {n_neg} muestras negativas.")

        return pd.DataFrame(X), pd.DataFrame(
            Y_binary, columns=["label"]
        )  # Devolvemos Y en formato binario

    def save_to_csv(self, df: pd.DataFrame, folder_path: str, filename: str) -> None:
        """Guarda el DataFrame como un archivo CSV en una carpeta específica"""

        # Especifica la ruta raíz de tu proyecto
        root_dir = r"C:\Users\Haison\Documents\movie_classifier"
        # Combina con la carpeta deseada
        full_folder_path = os.path.join(root_dir, folder_path)

        # Verifica si la carpeta existe
        if not os.path.exists(full_folder_path):
            # Si no existe, la crea
            os.makedirs(full_folder_path, exist_ok=True)
            logging.info(f"Carpeta creada: {full_folder_path}")
        else:
            logging.info(f"Carpeta existente: {full_folder_path}")

        # Crea la ruta completa del archivo
        full_path = os.path.join(full_folder_path, filename)

        # Guarda el DataFrame como CSV en la ruta especificada
        df.to_csv(full_path, index=False)
        print(f"Archivo guardado localmente como {full_path}")


# Ejecutar el código
if __name__ == "__main__":
    data = DataCleans3()
    datadf = data.read_s3_file("raw/ratings.dat")
    X, Y = data.transform_data(datadf)
    data.save_to_csv(X, "data/processed/", filename="X.csv")
    data.save_to_csv(Y, "data/processed/", filename="Y.csv")
