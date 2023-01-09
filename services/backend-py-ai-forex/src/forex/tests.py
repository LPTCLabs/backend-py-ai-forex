import pandas as pd
import constants
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/macro_data-all_data_2.0.csv", delimiter=",")
col = data.columns


# describe.to_csv("data/macro_data_describe.csv")

# Find outliers in all data using a box plot
# for column in data.columns:
#     if column != "date":
#         fig = px.box(data, y=column)
#         fig.update_layout(title_text="Box plot of " + column)
#         fig.show()


# clean outliers on specific columns
# def clean_outliers(column, data):
# q1 = data[column].quantile(0.25)
# q3 = data[column].quantile(0.75)
# iqr = q3 - q1
# lower = q1 - (1.5 * iqr).max()
# upper = q3 + (1.5 * iqr).min()
# data = np.where(data > upper, data.mean(), np.where(data < lower, data.mean(), data))

# box plot of the cleaned data
# fig = px.box(data, y=column)
# fig.update_layout(title_text="Box plot of " + column)
# fig.show()

# return data

# data_clean = clean_outliers("DE_Global_Construction_PMI", data)
# print(data_clean.describe())

# Convert the data to floats
# data = [float(x) for x in data]
#
# def minmax(data):
#     return (data - data.min()) / (data.max() - data.min())
#
#
# list_col = ["INR", "S&P500", "^NSEI"]
# x = data['Adj Close']
#
# plt.figure(figsize=(25, 10))
# plt.plot(minmax(x), 'k', lw=2, label='Adj Close')
#
# for col in list_col:
#     y = data[col]
#     plt.plot(minmax(y), label=col)
#
# plt.legend()
# plt.title("USDX vs INR")
# plt.grid()

# plot Adj Close and INR
# fig = px.line(data, x="Date", y=["INR", "Adj Close"])
# fig.update_layout(title_text="USDX vs INR")
# fig.show()
