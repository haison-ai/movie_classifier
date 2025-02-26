from numpy.matrixlib.defmatrix import matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

class MetricsModel:
    """Calculate metrics to measure fine and tunning the model"""
    def __init__(self, model_save = "models/PredictModel.pkl"):
        self.model_save = os.path.abspath(model_save)
        self.load_model()
        self.load_data()

    def load_data(self, X_path = r"C:\Users\Haison\Documents\movie_classifier\data\processed\X.csv", Y_path = r"C:\Users\Haison\Documents\movie_classifier\data\processed\Y.csv"):

        X_path = os.path.abspath(X_path)  # Convierte la ruta a absoluta
        Y_path = os.path.abspath(Y_path)

        if not os.path.exists(X_path) or not os.path.exists(Y_path):
            raise FileNotFoundError(f"No se encontró el archivo en {X_path} o {Y_path}")

        self.X1 = pd.read_csv(X_path).values
        self.Y1 = pd.read_csv(Y_path).values.ravel()

    def load_model(self):
        data = joblib.load(self.model_save)
        self.X_test = data["X_test"]
        self.Y_test = data["Y_test"]
        self.predictions = data["Predictions"]

    def matrix(self):
        matrix_confu = confusion_matrix(self.Y_test, self.predictions, labels=[0, 1])
        sns.heatmap(matrix_confu, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0",'Class 1'], yticklabels=["Class 0",'Class 1'])
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title("Confusion Matrix")
        plt.show()

    def metrics(self):
        report = classification_report(self.Y_test, self.predictions)
        return precision_score(self.Y_test, self.predictions, labels=[0, 1]), recall_score(self.Y_test, self.predictions, labels=[0, 1]), f1_score(self.Y_test, self.predictions, labels=[0, 1])


    def cross_val(self):
        k = 5
        k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        smoothing_factor_option = [1,2,3,4,5,6]
        fit_prior_option = [True, False]
        auc_record = {}
        for train_indices, test_indices in k_fold.split(self.X1, self.Y1):
            X_train_k, X_test_k = self.X1[train_indices], self.X1[test_indices]
            Y_train_k, Y_test_k = self.Y1[train_indices], self.Y1[test_indices]
            for alpha in smoothing_factor_option:
                if alpha not in auc_record:
                    auc_record[alpha] = {}
            for fit_prior in fit_prior_option:
                clf = MultinomialNB(alpha=alpha,fit_prior = fit_prior)
                clf.fit(X_train_k, Y_train_k)
                prediction_prob = clf.predict_proba(X_test_k)
                pos_prob = prediction_prob[:, 1]
                auc = roc_auc_score(Y_test_k, pos_prob)
                auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)
        for smoothing, smoothing_record in auc_record.items():
            for fit_prior, auc in smoothing_record.items():
                print(f' {smoothing} {fit_prior}{auc / k: .5f}')



"""

if __name__ == "__main__":
    metric = MetricsModel()
    print(metric.matrix())
    print(metric.metrics())
    print(metric.cross_val())
"""