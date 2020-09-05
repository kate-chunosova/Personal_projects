# Bike Sharing Model - Web App
*Author Ekaterina Chunosova*

## Overview

The `app.py` contains code for the flask application that can be run on its own. The rest of the files refer to the `ie_bike_model` package.

## Running the app

0. Make sure you have  `ie_bike_model` package installed, if not install it:  

```
$ pip install .
```

0. Make sure you have `flask` installed, otherwise:

```
$ pip install flask
```
1. To run the app type in the command line

```
$ flask run
```

## Usage

#### Getting the prediction
Launch the application and open http://127.0.0.1:5000/predict?date=2012-10-01T18:00:00&weathersit=1&temperature_C=15&feeling_temperature_C=14&humidity=20&windspeed=5&model=xgboost with your browser.

On this page user can see the model (Ridge or XGBoost by default) that the predictions were made with, prediction of the number of bikes and RMSE score.

**The input link can be modified in a couple of ways**:
- If the user wants to predict with Ridge he can specify `&model=ridge` in the link. If the parameter model is not provided at all, the default estimator is XGBoost (for example, http://127.0.0.1:5000/predict?date=2012-10-01T18:00:00&weathersit=1&temperature_C=15&feeling_temperature_C=14&humidity=20&windspeed=5)
- At all times the user must provide a date for the prediction, parameter `date=2012-10-01T18:00:00`, for example, is mandatory.
- The user might provide additional parameters for estimation, such as `weathersit`, `temperature_C`, `feeling_temperature_C`, `humidity` and `windspeed`. In case the user does not provide the input for any or all parameters, the prediction will be calculated on base of the mean values of the whole dataset and weathersit = 1 (the most frequent value). Thus, if the user doesn't know any characteristics of the day of the prediction he might use the link http://127.0.0.1:5000/predict?date=2012-10-01T18:00:00 which is going to calculate number of bikes on that day based on all mean parameters and xgboost as estimator.

#### Getting scores of the model
Launch the application and open http://127.0.0.1:5000/scores with your browser.

On this page the user can see the model (Ridge or XGBoost by default) that the predictions were made with, test and train R-squared.

**The input link can be modified in a couple of ways**:
- If the user wants to see the scores on the Ridge model, the link can be modified like so: http://127.0.0.1:5000/scores?model=ridge
- The user can also use the link http://127.0.0.1:5000/scores?model=xgboost to get the xgboost scores, however, it's easier to use just http://127.0.0.1:5000/scores since it will yield the same result.

**!** Ridge training time is about 5-10 seconds, whilst XGBoost training time is about 35-40 seconds. The first time the user clicks on the link the site is going to load for this amount of time (depending on the model).

## Advanced deployment

For example, with uWSGI:
```
$ pip install .
$ uwsgi --http 0.0.0.0:5000 --module ie_bike_web.app
```

Alternatively, to use the development server:
```
$ python -m ie_bike_web
```
For other options, check out the [Flask documentation.](https://flask.palletsprojects.com/en/1.1.x/deploying/)
