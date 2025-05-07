import joblib
import pandas as pd
import numpy as np
import os
import re

class PredictModel:
    """CLass to predict model already trained"""
    def __init__(self, model_load = "models/TrainModel.pkl", model_save = "models/PredictModel.pkl",X_path=r"C:\Users\Haison\Documents\movie_classifier\data\processed\X.csv", movies_path="C:/Users/Haison/Documents/movie_classifier/data/raw/movies.dat"):
        self.model_load = model_load
        self.model_save = model_save
        self.X_path = X_path
        self.movies_path = movies_path
        self.threshold = 0.1
        self.load_model()
        self.load_threshold()
        self.load_movies()

    def load_model(self):
        data = joblib.load(self.model_load)
        self.model = data["model"]
        self.X_test = data["X_test"]
        self.Y_test = data["Y_test"]

    def load_threshold(self):
        """Carga el threshold desde PredictModel.pkl"""
        if os.path.exists(self.model_save):
            data_thre = joblib.load(self.model_save)
            print(f" Contenido de PredictModel.pkl: {data_thre.keys()}")  # Verifica qué hay en el archivo

            self.threshold = data_thre.get("threshold", 0.5)  #  Usa 0.5 si no existe
            print(f" Threshold cargado correctamente: {self.threshold}")

            if self.threshold is None:
                print(" Warning: Threshold es None, asignando 0.5 por defecto.")
                self.threshold = 0.5
        else:
            self.threshold = 0.5  # Valor por defecto
            print(" Warning: PredictModel.pkl no encontrado. Usando threshold por defecto (0.5).")


    def load_movies(self):
        try:
            self.df_movies = pd.read_csv(self.movies_path, sep="::", engine="python", header=None, names=["MovieID", "Title", "Genres"], encoding="latin-1")
            print(self.df_movies.head(10))
        except:
            print("no se pudo cargar")
            self.df_movies = None

    def find_movie_id(self, review):
        if self.df_movies is None:
            print(" Movie data not loaded")
            return None

        review = review.lower().strip()
        best_match = None
        best_score = 0

        for _, row in self.df_movies.iterrows():
            title = row["Title"].lower().strip()
            title_clean = re.sub(r"\(\d{4}\)", "", title).strip()

            if title_clean in review:
                match_score = len(title_clean) / len(review)  # Relación de longitud
                if match_score > best_score:
                    best_match = (row["MovieID"], title_clean)
                    best_score = match_score

        if best_match:
            movie_id, best_title = best_match
            print(f" Found: {best_title} (MovieID: {movie_id})")
            return movie_id

        print(" No movie found")
        return None

    def obtain_vector(self, movie_id):
        if not os.path.exists(self.X_path):
            print(f"No file")
            return None


        try:
            X = pd.read_csv(self.X_path, header=None)
        except FileNotFoundError:
            print("No file found")
            return None

        if 0 <= movie_id < len(X):
            vector = X.iloc[movie_id].values.reshape(1, -1)

            if isinstance(self.X_test, pd.DataFrame):
                vector_df = pd.DataFrame(vector, columns=self.X_test.columns)
            else:
                vector_df = pd.DataFrame(vector)

            return vector_df
        else:
            print("Movie not found")
            return None

    def predict_review(self, review):
        print(f" Threshold antes de predecir: {self.threshold}")
        movie_id = self.find_movie_id(review)

        if movie_id is not None:
            X_new = self.obtain_vector(movie_id)

            if X_new is not None:
                predict_prob = self.model.predict_proba(X_new)
                pro_class_1 = predict_prob[0][1]

                print(f"treshold {self.threshold}")
                print(f"probibilidad {pro_class_1}")
                predict_r = 1 if pro_class_1 > self.threshold else 0
                result = " Recommended" if predict_r == 1 else "❌ Not recommended"
                print(f"Recommendation: {result} for {movie_id}")
                print(f"Predicted proba for {predict_prob}")
                return predict_r

        print("Prediction failed")
        return None



    def predict_proba(self):
        proba = self.model.predict_proba(self.X_test)[:,1]
        return proba

    def predict(self):
        predictions = self.model.predict(self.X_test)
        return predictions

    def accuracy(self):
        acurracy = self.model.score(self.X_test, self.Y_test)
        return acurracy

    def save_model(self, threshold=None):
        self.threshold = threshold if threshold is not None else self.threshold
        predictions = self.predict()
        proba = self.predict_proba()
        accuracy = self.accuracy()

        results = {
            "model": self.model,
            "X_test": self.X_test,
            "Y_test": self.Y_test,
            "Predictions": predictions,
            "Predictions_proba": proba,
            "Accuracy": accuracy,
            "threshold": self.threshold
        }

        os.makedirs(os.path.dirname(self.model_save), exist_ok=True)
        joblib.dump(results, self.model_save)
        print(f"Model saved")




if __name__ == "__main__":
    predictor = PredictModel()
    prob = predictor.predict_proba()
    pred = predictor.predict()
    acur = predictor.accuracy()
    predictor.save_model(threshold=0.05)
    print(prob[0:10])
    print(pred[:10])
    print(f"The acurracy is around: {acur*100:.1f}%")

    review_usuario = input("Write a review: ")
    predictor.predict_review(review_usuario)
    predictor.load_threshold()


    data = joblib.load("models/PredictModel.pkl")
    print(data.keys())  # Verifica si "threshold" está presente
    print(f"Threshold guardado: {data.get('threshold')}")






