from typing import Any

import pandas as pd
from numpy import mean
from numpy import std
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from . import constants


def train(X: list, Y: list) -> Any:
    if constants.MODEL_TYPE == "logistic":
        # create model
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",  # ‘lbfgs’, ‘newton-cg’, ‘sag’, ‘saga’
            max_iter=10000,
        )
    elif constants.MODEL_TYPE == "SVM":
        model = svm.SVC(kernel="linear", max_iter=10000, probability=True)
    elif constants.MODEL_TYPE == "random_forest":
        model = RandomForestClassifier(n_estimators=100)
    elif constants.MODEL_TYPE == "decision_tree":
        model = DecisionTreeClassifier()
    else:
        raise Exception("Invalid model type")
    model.fit(X, Y)
    return model

# def train(X: list, Y: list) -> Any:
#     if constants.MODEL_TYPE == "logistic":
#         model = LogisticRegression()
#         hyperparameters = {
#             "C": [300, 600, 900],
#             "penalty": ["l2"],  # l1, l2, elasticnet
#             "solver": ["lbfgs"],  # saga, liblinear, lbfgs, newton-cg, sag
#             "max_iter": [10000],
#         },
#         gs = GridSearchCV(model, hyperparameters, scoring="accuracy", cv=5, verbose=0)
#     else:
#         constants.MODEL_TYPE == "SVM"
#         model = svm.SVC()
#         hyperparameters = {
#             "kernel": ["linear"],  # rbf, poly, sigmoid
#             "C": [100, 500, 1000],
#             "max_iter": [10000],
#         },
#         gs = GridSearchCV(model, hyperparameters, scoring='accuracy', cv=5, verbose=0)
#     gs.fit(X, Y)
#     best_estimator = gs.best_estimator_
#     best_param = gs.best_params_
#     best_score = gs.best_score_
#     df = pd.DataFrame(gs.cv_results_)
#     df.to_csv('grid_search_results.csv')
#     return gs, best_estimator, best_param, best_score


def generate_predictions(model: Any, test_input_data: list) -> tuple:
    """
    :param model:
    :param test_input_data:
    :return:
    """
    all_probs: list = []
    for input_data in test_input_data:
        if constants.MODEL_TYPE == "logistic":
            probas = model.predict_proba([input_data]).tolist()[0]
        elif constants.MODEL_TYPE == "SVM":
            probas = model.predict_proba([input_data]).tolist()[0]
        elif constants.MODEL_TYPE == "random_forest":
            probas = model.predict_proba([input_data]).tolist()[0]
        elif constants.MODEL_TYPE == "decision_tree":
            probas = model.predict_proba([input_data]).tolist()[0]
        else:
            raise Exception("Invalid model type")
        all_probs.append(probas)
    predictions: list = []
    sorted_predictions: list = []
    for example in all_probs:
        _zipped = list(zip(example, model.classes_))
        _sorted = sorted(_zipped, reverse=True)
        predictions.append([x[:] for x in [_zipped][0]])
        sorted_predictions.append([x[:] for x in [_sorted][0]])
    return predictions, sorted_predictions
