import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class TrainModel:
    """Class to train the model"""
    def __init__(self, model_path="models/TrainModel.pkl"):
        self.model_path = model_path
        self.model = MultinomialNB(alpha=1.0, fit_prior=True)

    def load_data(
        self,
        X_path=r"C:\Users\Haison\Documents\movie_classifier\data\processed\X.csv",
        Y_path=r"C:\Users\Haison\Documents\movie_classifier\data\processed\Y.csv",
    ):
        self.X = pd.read_csv(X_path)
        self.Y = pd.read_csv(Y_path).values.ravel()

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state
        )
        print(len(self.Y_train), len(self.Y_test))

    def train(self):
        self.model.fit(self.X_train, self.Y_train)

    def save_model(self):
        result = {"model": self.model, "X_test": self.X_test, "Y_test": self.Y_test}
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(result, self.model_path)

    def run_pipeline(self):
        self.load_data()
        self.split_data()
        self.train()
        self.save_model()

"""
if __name__ == "__main__":
    trainer = TrainModel()
    trainer.run_pipeline()
"""