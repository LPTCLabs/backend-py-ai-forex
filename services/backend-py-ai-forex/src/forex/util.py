import csv

from scipy import stats
from sklearn import metrics
import pandas as pd
import numpy
import matplotlib.pyplot
import os
import json

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import constants


def load_macro_data_wo_delay(source: str) -> tuple:
    """
    Load macro data
    :param source:
    :return: Dictionary with macro data for all years.
    """
    data: dict = {}
    usd: dict = {}
    with open(source, newline="") as csvfile:
        content = csv.DictReader(csvfile, delimiter=",")
        previous_line: list = []
        for i, line in enumerate(content):
            values: list = []
            total: int = 0
            # US_EA_BINARY_FEATURES CLASSIFICATION
            if constants.US_EA_CLASSIFICATION:
                if constants.ADD_MACRO:
                    for feature in constants.US_EA_BINARY_FEATURES["MACRO"]:
                        if feature in line:
                            total += 1
                            values.append(line[feature])
                    if constants.ADD_COMMODITY:
                        for feature in constants.US_EA_BINARY_FEATURES["COMMODITY"]:
                            if feature in line:
                                total += 1
                                values.append(line[feature])
                    if constants.ADD_FI:
                        for feature in constants.US_EA_BINARY_FEATURES["FI"]:
                            if feature in line:
                                total += 1
                                values.append(line[feature])
                    if constants.ADD_EQUITY:
                        for feature in constants.US_EA_BINARY_FEATURES["EQUITY"]:
                            if feature in line:
                                total += 1
                                values.append(line[feature])
            elif constants.TEST:
                if constants.ADD_MACRO:
                    for feature in constants.US_EA_BINARY_FEATURES2["MACRO"]:
                        if feature in line:
                            total += 1
                            values.append(line[feature])
                    if constants.ADD_COMMODITY:
                        for feature in constants.US_EA_BINARY_FEATURES2["COMMODITY"]:
                            if feature in line:
                                total += 1
                                values.append(line[feature])
                    if constants.ADD_FI:
                        for feature in constants.US_EA_BINARY_FEATURES2["FI"]:
                            if feature in line:
                                total += 1
                                values.append(line[feature])
                    if constants.ADD_EQUITY:
                        for feature in constants.US_EA_BINARY_FEATURES2["EQUITY"]:
                            if feature in line:
                                total += 1
                                values.append(line[feature])
            elif constants.BINARY_CLASSIFICATION:
                if constants.ADD_MACRO:
                    for feature in constants.US_EA_BINARY_FEATURES2["MACRO"]:
                        if feature in line:
                            total += 1
                            values.append(line[feature])
                    if constants.ADD_COMMODITY:
                        for feature in constants.US_EA_BINARY_FEATURES2["COMMODITY"]:
                            if feature in line:
                                total += 1
                                values.append(line[feature])
                    if constants.ADD_FI:
                        for feature in constants.US_EA_BINARY_FEATURES2["FI"]:
                            if feature in line:
                                total += 1
                                values.append(line[feature])
                    if constants.ADD_EQUITY:
                        for feature in constants.US_EA_BINARY_FEATURES2["EQUITY"]:
                            if feature in line:
                                total += 1
                                values.append(line[feature])
            else:
                raise Exception("Invalid test selected")
            constants.NB_MACRO_DATA = total
            date = line["Date"][:-3]
            usd[date] = float(line["Adj Close"])
            data[date] = values
            if len(previous_line) > 0:
                for _pos in range(len(previous_line)):
                    if len(data[date][_pos]) == 0 and len(previous_line[_pos]) > 0:
                        data[date][_pos] = previous_line[_pos]
                    elif (
                            len(data[date][_pos]) == 0 and len(previous_line[_pos]) == "''"
                    ):
                        data[date][_pos] = "0.0"
            else:
                for _pos in range(len(values)):
                    if len(data[date][_pos]) == 0:
                        data[date][_pos] = "0.0"

            previous_line = data[date][:]
    return data, usd


def load_macro_dataset(source: str, macro_data: dict) -> list:
    """
    Load inflation and growth dataset.
    :param source: Source for the inflation and growth data.
    :param macro_data: Macro data
    :return: Dictionary with all growth, inflation and macro data.
    """
    dataset = []
    with open(source, newline="") as csvfile:
        # fieldnames = ['region_code', 'updated_date', 'growth_raw_value', 'growth_moving_average_value',
        #              'growth_exponential_moving_average_value', 'Growth_CrossOver1', 'inflation_raw_value',
        #              'inflation_moving_average_value', 'inflation_exponential_moving_average_value',
        #              'Inflation_CrossOver1', 'growth_CrossOver_Prod', 'inflation_CrossOver_Prod', 'Regime(redefined)']
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = row["updated_date"].split("/")
            month = date[2] + "-" + str(date[0]).zfill(2)
            values = [
                row["updated_date"],
                row["Regime(redefined)"].lower(),
                float(row["growth_raw_value"].replace(",", "")),
                float(row["inflation_raw_value"].replace(",", "")),
            ]
            if constants.ADD_MOVING_AVERAGE:
                values.extend(
                    [
                        float(row["growth_moving_average_value"].replace(",", "")),
                        float(row["inflation_moving_average_value"].replace(",", "")),
                    ]
                )
            if month in macro_data:
                values.extend(macro_data[month])
            else:
                values.extend([0.0] * constants.NB_MACRO_DATA)
            dataset.append(values)
    return dataset


def split_input_output_dataset(dataset: list, usd: dict) -> list:
    """
    Split dataset in input and output data.
    :param dataset: Dataset
    :return: Split dataset
    """
    split_dataset = []
    window = constants.WINDOW_PROBABILITY_ANALYSIS_LONGEST
    for i, values in enumerate(dataset):
        if window <= i:
            _input = dataset[i - window: i]
            # _output = dataset[i][1]
            # print(_input)
            # print(dataset[i])
            month, day, year = dataset[i - 1][0].split("/")
            previous_month = f"{year}-{int(month):02d}"
            month, day, year = dataset[i][0].split("/")
            predict_month = f"{year}-{int(month):02d}"
            if previous_month not in usd or predict_month not in usd:
                continue
            # BINARY CLASSIFICATION
            if usd[predict_month] - usd[previous_month] > constants.BINARY_THRESHOLD_FOREX:
                _output = "up"
            else:
                _output = "down"
            split_dataset.append([_input, _output, dataset[i][0]])
    return split_dataset


def calculate_probability_inflation_growth(values: list) -> dict:
    """
    Calculate the probability for all inflation and growth situation.
    :param values:
    :return:
    """
    inf_down, inf_up, growth_down, growth_up = 0.0, 0.0, 0.0, 0.0
    avg_growth_slope, avg_inf_slope = 0.0, 0.0
    total = 0.0
    if constants.WINDOW_PROBABILITY_ANALYSIS_LONGEST:
        for i in range(constants.WINDOW_PROBABILITY_ANALYSIS_NUMBER):
            x = list(range(i, len(values)))
            gro = [values[ii][2] for ii in range(i, len(values))]
            inf = [values[ii][3] for ii in range(i, len(values))]
            inf_slope, _, _, _, _ = stats.linregress(x, inf)
            gro_slope, _, _, _, _ = stats.linregress(x, gro)

            avg_growth_slope += gro_slope
            avg_inf_slope += inf_slope

            if gro_slope > 0.01:
                growth_up += 1.0
            else:
                growth_down += 1.0
            if inf_slope > 0.01:
                inf_up += 1.0
            else:
                inf_down += 1.0
            total += 1.0
        probs: dict = {
            "inflationdown_growthdown": (inf_down / total) * (growth_down / total),
            "inflationdown_growthup": (inf_down / total) * (growth_up / total),
            "inflationup_growthdown": (inf_up / total) * (growth_down / total),
            "inflationup_growthup": (inf_up / total) * (growth_up / total),
        }
    else:
        for i in range(constants.WINDOW_PROBABILITY_ANALYSIS_LONGEST):
            x = list(range(i, len(values)))
        gro = [values[ii][2] for ii in range(i, len(values))]
        inf = [values[ii][3] for ii in range(i, len(values))]
        inf_slope, _, _, _, _ = stats.linregress(x, inf)
        gro_slope, _, _, _, _ = stats.linregress(x, gro)

        avg_growth_slope += gro_slope
        avg_inf_slope += inf_slope

        if gro_slope > 0.01:
            growth_up += 1.0
        else:
            growth_down += 1.0
        if inf_slope > 0.01:
            inf_up += 1.0
        else:
            inf_down += 1.0
        total += 1.0
    probs: dict = {
        "inflationdown_growthdown": (inf_down / total) * (growth_down / total),
        "inflationdown_growthup": (inf_down / total) * (growth_up / total),
        "inflationup_growthdown": (inf_up / total) * (growth_down / total),
        "inflationup_growthup": (inf_up / total) * (growth_up / total),
    }
    return probs


def keep_unrepeated_values(values: list) -> list:
    """
    Delete sequence of repeated values in a list.
    :param values: List of values.
    :return:
    """
    new_list: list = [values[0]]
    for i in range(1, len(values)):
        if values[i] != new_list[-1]:
            new_list.append(values[i])
    return new_list


def calculate_average_slope(values: list, _index: int) -> float:
    """
    Calculate the probability for all inflation and growth situation.
    :param _index:
    :param values:
    :return:
    """
    slope_total = 0.0
    total = 0.0
    for i in range(constants.WINDOW_PROBABILITY_ANALYSIS_NUMBER):
        _values = [float(values[ii][_index]) for ii in range(i, len(values))]
        _values = keep_unrepeated_values(_values)
        x = list(range(len(_values)))
        if len(_values) > 1:
            slope, _, _, _, _ = stats.linregress(x, _values)
        else:
            slope = 0.0
        slope_total += slope
        total += 1.0
    avg_slope = numpy.tanh(slope_total / total)
    return avg_slope


def get_X_Y(data: list) -> tuple:
    """
    Generate input (X) and output (Y) to train the economic regime model.
    :param data: List with data
    :return:
    """
    detail: dict = {}
    for i in range(len(data)):
        label = data[i][0][-1][1] + "-" + data[i][1]
        if label not in detail:
            detail[label] = []
        values = data[i][0]
        probs = calculate_probability_inflation_growth(values)
        _input = [
            probs["inflationdown_growthdown"],
            probs["inflationdown_growthup"],
            probs["inflationup_growthdown"],
            probs["inflationup_growthup"],
        ]
        for _index in range(2, len(values[0])):
            avg_slope = calculate_average_slope(values, _index)
            _input.append(avg_slope)
        detail[label].append((_input, data[i][1], data[i][0]))
    input_data, output_data, dates = [], [], []
    for label in detail:
        input_data.extend([data[0] for data in detail[label][:]])
        output_data.extend([data[1] for data in detail[label][:]])
        dates.extend([data[2] for data in detail[label][:]])
    return input_data, output_data, dates


def get_X_Y_test(data: list) -> tuple:
    """
    Generate input (X) and output (Y) to test the economic regime model.
    :param data: List with data
    :return:
    """
    input_data, output_data, dates = [], [], []
    for i in range(len(data)):
        values = data[i][0]
        probs = calculate_probability_inflation_growth(values)
        _input = [
            probs["inflationdown_growthdown"],
            probs["inflationdown_growthup"],
            probs["inflationup_growthdown"],
            probs["inflationup_growthup"],
        ]
        for _index in range(2, len(values[0])):
            avg_slope = calculate_average_slope(values, _index)
            _input.append(avg_slope)
        input_data.append(_input)
        output_data.append(data[i][1])
        dates.append(data[i][2])
    # print('Total examples:', len(input_data))
    return input_data, output_data, dates


def compute_accuracy(Y_test: list, predictions: list) -> float:
    """
    Compute the accuracy of model.
    :param Y_test: Gold values
    :param predictions: Prediction of the model
    :return:
    """
    return metrics.accuracy_score(Y_test, predictions)


def write_json_csv(filename, json_str, append=False):
    path = f"{filename}.csv"
    write_header = not os.path.exists(path) or append is False
    with open(path, "a+" if append else "w", encoding="UTF-8") as csv_file:
        csv_writer = csv.writer(csv_file, lineterminator="\n")
        json_data = json.loads(json_str)
        if json_data:
            if write_header:
                try:
                    csv_writer.writerow(json_data.keys())
                except AttributeError:
                    csv_writer.writerow(json_data[0].keys())
            try:
                csv_writer.writerow(json_data.values())
            except AttributeError:
                for row in json_data:
                    csv_writer.writerow(row.values())


def plot_columns(data: list, indexed_by, columns: list, colors: list) -> matplotlib.pyplot:
    """
    Plot the columns on the same graph.
    :param data: List with data
    :param indexed_by: name of indexed column
    :param columns: List of columns to plot
    :param colors: List of colors for each column
    :return:
    """
    df = pd.DataFrame(data)
    matplotlib.pyplot.figure(figsize=(20, 10))
    matplotlib.pyplot.title(" ".join(columns))
    for column in columns:
        matplotlib.pyplot.plot(df[indexed_by], df[column])  # , color=(colors[i] or None))
        matplotlib.pyplot.xlabel(indexed_by)
        matplotlib.pyplot.ylabel(column)

    return matplotlib.pyplot


def generate_dataset(
        slash_dates: list, prices: pd.Series, goal: list, predicted: list, months: tuple = (), threshold: float = 1.0
) -> list:
    threshold_variation = 0.0
    if months:
        months_index = [i for i in range(len(slash_dates)) if slash_dates[i].startswith(months)]
        slash_dates = [x for i, x in enumerate(slash_dates) if i in months_index]
        goal = [x for i, x in enumerate(goal) if i in months_index]
        predicted = [x for i, x in enumerate(predicted) if i in months_index]
    previous_date = ""
    combined = []
    for i in range(len(slash_dates)):
        month, day, year = slash_dates[i].split("/")
        date = f"{year}-{int(month):02d}"
        price_variation: float = 0.0 if i == 0 else prices[previous_date] - prices[date]
        previous_date = date
        if predicted[i] == "down":
            threshold_variation -= threshold
        else:
            threshold_variation += threshold
        current = {
            "date": date,
            "orientation": 0 if goal[i] == "down" else 1,
            "prediction": 0 if predicted[i] == "down" else 1,
            "difference": 0 if (i == 0 or (i > 0 and goal[i] == predicted[i])) else 1,
            "price": prices[date],
            "reversed_price_variation": -price_variation,  # signed for reversal
            "threshold_variation": threshold_variation,
        }
        combined.append(current)

    return combined


def vp(usd: pd.Series, predicted_labels, test_dates) -> pd.Series:
    """
    Plot usd variation one data every 6 months
    :param usd:
    :param predicted_labels:
    :param test_dates:
    :return:
    """
    # keep only values from usd dict
    usd = [usd[date] for date in usd]
    # keep same len of data as prediction
    usd = usd[: len(predicted_labels)]
    # calculate usd variation
    usd = [usd[i] - usd[i - 1] for i in range(1, len(usd))]
    matplotlib.pyplot.figure(figsize=(20, 10))
    if constants.PLOT_VP_SIX_MONTHS:
        matplotlib.pyplot.title("Dollar variation vs predictions - Every 6 months")
        matplotlib.pyplot.plot(test_dates[::6], usd[::6], color="blue")
        matplotlib.pyplot.plot(test_dates[::6], predicted_labels[::6], color="red")
    else:
        matplotlib.pyplot.title("Dollar variation vs predictions - All data")
        matplotlib.pyplot.plot(test_dates, usd, color="blue")
        matplotlib.pyplot.plot(test_dates, predicted_labels, color="red")
    matplotlib.pyplot.xticks(rotation=45)
    matplotlib.pyplot.yticks(["down", "up"])
    return matplotlib.pyplot

