import os
import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split




class train_model:
    def __init__(self, model_path = "models/movie_classifier.pkl"):
        self.model_path = model_path
        self.model = MultinomialNB()

    def load_data(self, X_path = r"C:\Users\Haison\Documents\movie_classifier\data\processed\X.csv", Y_path = r"C:\Users\Haison\Documents\movie_classifier\data\processed\Y.csv"):
        self.X = pd.read_csv(X_path)
        self.Y = pd.read_csv(Y_path).values.ravel()


    def split_data(self, test_size = 0.2, random_state = 42):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size = test_size, random_state = random_state)
        print(len(self.Y_train), len(self.Y_test))

    def train(self):
        self.model.fit(self.X_train, self.Y_train)

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def run_pipeline(self):
        self.load_data()
        self.split_data()
        self.train()
        self.save_model()


if __name__ == '__main__':
    trainer = train_model()
    trainer.run_pipeline()