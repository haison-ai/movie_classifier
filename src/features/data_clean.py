import pandas as pd
import numpy as np
import boto3
import logging
from io import BytesIO
from src.features.load_rating import load_user_rating_data
#from .data import loader_s3


class DataCleans3:
    def __init__(self, s3_name="movieclassifiers3", region="us-east-2"):
        "Start client with aws s3"
        self.s3_name = s3_name
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)

    def read_s3_file(self, s3_route) -> pd.DataFrame:
        "read file from s3"

        try:
            csv = self.s3.get_object(Bucket=self.s3_name, Key="raw/ratings.dat")
            df = pd.read_csv(
                BytesIO(csv["Body"].read()),  delimiter="::", engine="python"
            )
            logging.info("file load successfully from s3")
            return df

        except Exception as e:
            logging.error("Error to read file from s3: {e}")
            return None

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = ["user_id", "movie_id", "rating", "timestamp"]
        print(df.head())
        n_users = df["user_id"].nunique()
        n_movies = df["movie_id"].nunique()
        print(f"number of users: {n_users}, ",f"number of movies: {n_movies}")




if __name__ == "__main__":
    data = DataCleans3()
    df_row = data.transform_data(data.read_s3_file("raw/ratings.dat"))

    # loads3 = loader_s3() # class to load dataFrame to s3
    # loads3.load_s3("data/proccesed/")