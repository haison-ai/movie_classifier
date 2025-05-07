from src.models.train_model import TrainModel
from src.models.predict_model import PredictModel
from src.models.metrics_model import MetricsModel

def main():

    "Train model and save with joblib the trained file "
    print(" Training model....")
    model = TrainModel()
    model.run_pipeline()

    #Train model and save with joblib the trained file
    print(" Predicting model....")
    predict = PredictModel()
    pred = predict.predict()
    predict.accuracy()
    print(pred)


    print(" Metrics model....")
    metrics = MetricsModel()
    print(metrics.metrics())
    metrics.cross_val()
    metrics.matrix()


if __name__ == "__main__":
    main()
