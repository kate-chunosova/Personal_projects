import datetime as dt
import numpy as np
import joblib

from flask import Flask, request

from ie_bike_model.util import read_data
from ie_bike_model.model import predict, score

app = Flask(__name__)


@app.route("/")
def hello():
    name = request.args.get("name", "World")
    return "Hello, " + name + "!"


@app.route("/predict", methods=["GET"], endpoint="get_predict")
def get_predict():

    parameters = dict()
    parameters["date"] = dt.datetime.fromisoformat(request.args.get("date"))
    parameters["weathersit"] = int(request.args.get("weathersit"))
    parameters["temperature_C"] = float(request.args.get("temperature_C"))
    parameters["feeling_temperature_C"] = float(
        request.args.get("feeling_temperature_C")
    )
    parameters["humidity"] = float(request.args.get("humidity"))
    parameters["windspeed"] = float(request.args.get("windspeed"))
    model = str(request.args.get("model", "xgboost"))

    result = predict(parameters, model=model)

    # Return RMSE
    hour_test = read_data().iloc[15212:17379]
    hour_test_y = hour_test["cnt"]
    RMSE = np.sqrt(np.mean((hour_test_y * 2 - result * 2) ** 2))

    return {"Model": model, "Number of bikes": result, "Test RMSE": RMSE.round(3)}


@app.route("/scores", methods=["GET"], endpoint="get_score")
def get_score():
    model = request.args.get("model", "xgboost")
    r2_train, r2_test = score(model=model)

    return {
        "R-squared for Train:": r2_train.round(2),
        "R-squared for Test:": r2_test.round(2),
    }
