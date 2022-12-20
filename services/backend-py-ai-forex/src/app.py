import csv
import json
import logging
from typing import Any
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter

from .forex import constants, machine_learning, util


def predict_dataset(model: Any, source: str, macro_data: dict, usd: pd.Series) -> None:
    """
    Predict the economic regime for a dataset.
    :param model: Economic regime model
    :param source: Source of data
    :param macro_data: Macro data
    :param usd: Dollar index
    :return:
    """
    # Load test data
    test_dataset = util.load_macro_dataset(source, macro_data)
    test_data = util.split_input_output_dataset(test_dataset, usd)

    # Predict the test data
    test_input_data, gold_test, test_dates = util.get_X_Y_test(test_data)
    predictions, sorted_predictions = machine_learning.generate_predictions(model, test_input_data)

    predicted_labels = [prediction[0][1] for prediction in sorted_predictions]

    # combined = util.generate_dataset(test_dates, usd, gold_test, predicted_labels, threshold=0.1)  # monthly
    # combined = util.generate_dataset(test_dates, usd, gold_test, predicted_labels, months=("6", "12"))  # bi_yearly

    # price_variation_plot = util.plot_columns(combined, "date", ["threshold_variation", "reversed_price_variation"], ["g", "r"])
    # useful to identify areas that were wrong or not
    # for i, x in enumerate(combined):
    #     price_variation_plot.axvspan(i, i + 1.0, facecolor="g" if x["difference"] == 0 else "r", alpha=0.2)
    # price_variation_plot.legend(["prediction", "price_variation"])
    # price_variation_plot.savefig("price_variation.png")
    # price_variation_plot.show()
    #
    # variation_plot = util.plot_columns(combined, "date", ["orientation", "prediction", "difference"], ["g", "r", "b"])
    # variation_plot.legend(["prediction", "reversed_price_variation", "difference"])
    # variation_plot.savefig("variation_prediction.png")
    # variation_plot.show()
    #
    # difference_plot = util.plot_columns(combined, "date", ["difference"], ["b"])
    # difference_plot.legend(["prediction", "reversed_price_variation"])
    # difference_plot.savefig("difference.png")
    # difference_plot.show()

    # variation_prediction = util.vp(usd, predicted_labels, test_dates)
    # variation_prediction.legend(["Dollar variation", "Predictions"])
    # variation_prediction.savefig("new_vp.png")
    # variation_prediction.show()

    # util.write_json_csv("variation_prediction", json.dumps(combined))
    accuracy = util.compute_accuracy(gold_test, predicted_labels)

    # Accuracy metrics
    if constants.US_EA_CLASSIFICATION:
        print("Test 1 : US-EA Classification")
        print(f"\nComplete accuracy : {accuracy}")
        # Generate a confusion matrix
        matrix = metrics.confusion_matrix(gold_test, predicted_labels, labels=constants.BINARY_LABELS)
        matrix_df = pd.DataFrame(matrix, index=constants.BINARY_LABELS, columns=constants.BINARY_LABELS)
        print("#########")
        print("### Confusion matrix\n#########\n", matrix_df)
        print("\n#########")
        print("### Classification Report\n#########\n", metrics.classification_report(gold_test, predicted_labels))
    elif constants.TEST:
        print("Test 2 : using different features")
        print(f"\nComplete accuracy : {accuracy}")
        matrix = metrics.confusion_matrix(gold_test, predicted_labels, labels=constants.BINARY_LABELS)
        matrix_df = pd.DataFrame(matrix, index=constants.BINARY_LABELS, columns=constants.BINARY_LABELS)
        print("#########")
        print("### Confusion matrix\n#########\n", matrix_df)
        print("\n#########")
        print("### Classification Report\n#########\n", metrics.classification_report(gold_test, predicted_labels))
    elif constants.BINARY_CLASSIFICATION:
        print("Test 3 : US classification")
        print(f"\nComplete accuracy : {accuracy}")
        matrix = metrics.confusion_matrix(gold_test, predicted_labels, labels=constants.BINARY_LABELS)
        matrix_df = pd.DataFrame(matrix, index=constants.BINARY_LABELS, columns=constants.BINARY_LABELS)
        print("#########")
        print("### Confusion matrix\n#########\n", matrix_df)
        print("\n#########")
        print("### Classification Report\n#########\n", metrics.classification_report(gold_test, predicted_labels))
    else:
        raise Exception("Invalid test selected")

    # Features automating selection
    # constants_features = constants.US_EA_BINARY_FEATURES.copy()
    # with open("results.txt", "a") as f:
    #     constants.FEATURES = []
    #     for model in ["logistic"]:
    #         # for one in tqdm(range(5, 13)):
    #         #     constants.WINDOW_PROBABILITY_ANALYSIS_LONGEST = one
    #         #     for two in range(5, 13):
    #         #         constants.WINDOW_PROBABILITY_ANALYSIS_NUMBER = two
    #         #         for three in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #         #             constants.BINARY_THRESHOLD_FOREX = three
    #         constants.US_EA_BINARY_FEATURES["MACRO"] = constants.KEEP_MACRO.copy()
    #         for macro in constants_features["MACRO"]:
    #             if macro in constants.KEEP_MACRO:
    #                 constants.US_EA_BINARY_FEATURES["MACRO"].append(macro)
    #                 while constants.KEEP_COMMODITY and constants.LOSE_COMMODITY is True:
    #                     for commodity in constants_features["COMMODITY"]:
    #                         if constants_features["COMMODITY"] <= constants.MIN_COMMODITY:
    #                             continue
    #                         constants_features["COMMODITY"] = commodity
    #                         for fi in constants_features["FI"]:
    #                             if constants_features["FI"] <= constants.MIN_FI:
    #                                 continue
    #                             constants_features["FI"] = fi
    #                             for equity in constants_features["EQUITY"]:
    #                                 if constants_features["EQUITY"] <= constants.MIN_EQUITY:
    #                                     continue
    #                                 constants_features["EQUITY"] = equity
    #                                 constants.MODEL_TYPE = model
    #                                 logging.info("---------------------------\n"
    #                                              # f"Longest: {one} - Number: {two} - Threshold: {three}\n"
    #                                              f"MACRO: {macro} - COMMODITY: {commodity} - FI: {fi}\n"
    #                                              f"EQUITY: {equity} - MODEL: {model}\n"
    #                                              f"Accuracy : {accuracy}\n"
    #                                              "---------------------------")
    #                                 df = pd.DataFrame([[
    #                                         # one,
    #                                         # two,
    #                                         # three,
    #                                         macro,
    #                                         commodity,
    #                                         fi,
    #                                         equity,
    #                                         model,
    #                                         accuracy
    #                                 ]],
    #
    #                                     columns=[
    #                                         # "longest",
    #                                         # "number",
    #                                         # "threshold",
    #                                         "macro",
    #                                         "commodity",
    #                                         "fi",
    #                                         "equity",
    #                                         "model",
    #                                         "accuracy"
    #                                     ])
    #                                 df.to_csv("results.csv", mode="a", header=False)


def train_model() -> None:
    # Load macro data
    macro_data, usd = util.load_macro_data_wo_delay(constants.MACRO_DATA)
    if constants.TRAINING_WITH_TRAIN_DEV_DATA:
        # Train model using training + dev data
        train_dataset = util.load_macro_dataset(constants.TRAIN_DEV_DATA, macro_data)
    else:
        # Train model using training data
        train_dataset = util.load_macro_dataset(constants.TRAIN_DATA, macro_data)
    train_data = util.split_input_output_dataset(train_dataset, usd)
    # Train model using training dataset
    train_input_data, train_output_data, train_dates = util.get_X_Y(train_data)
    print("Value Count Train Output Data", Counter(train_output_data))
    # Train model
    model = machine_learning.train(train_input_data, train_output_data)
    # Predict on test data
    predict_dataset(model, constants.TEST_DATA, macro_data, usd)


# def lunch_model() -> None:
#     util.automate_randomly_the_process(train_model())


if __name__ == "__main__":
    # lunch_model()
    train_model()
