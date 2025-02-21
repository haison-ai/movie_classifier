import joblib
import pandas as pd
import os

class PredictModel:
    def __init__(self, model_load = "models/TrainModel.pkl", model_save = "models/PredictModel.pkl"):
        self.model_load = model_load
        self.model_save = model_save
        self.load_model()

    def load_model(self):
        data = joblib.load(self.model_load)
        self.model = data["model"]
        self.X_test = data["X_test"]
        self.Y_test = data["Y_test"]

    def predict_proba(self):
        proba = self.model.predict_proba(self.X_test)
        return proba

    def predict(self):
        predictions = self.model.predict(self.X_test)
        return predictions

    def accuracy(self):
        acurracy = self.model.score(self.X_test, self.Y_test)
        return acurracy

    def save_model(self):
        predictions = self.predict()
        proba = self.predict_proba()
        accuracy = self.accuracy()

        results = {
            "X_test": self.X_test,
            "Y_test": self.Y_test,
            "Predictions": predictions,
            "Probabilities": proba,
            "Accuracy": accuracy
        }

        os.makedirs(os.path.dirname(self.model_save), exist_ok=True)
        joblib.dump(results, self.model_save)


if __name__ == "__main__":
    predictor = PredictModel()
    prob = predictor.predict_proba()
    pred = predictor.predict()
    acur = predictor.accuracy()
    predictor.save_model()
    print(prob[0:10])
    print(pred[:10])
    print(f"The acurracy is around: {acur*100:.1f}%")




