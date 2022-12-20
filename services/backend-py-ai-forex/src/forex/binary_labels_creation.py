import pandas as pd
import constants

data = pd.read_csv("data/macro_data-all_data_2.0.csv")
print(data)

def binary_labels_creation(df):
    """
    :label calculation : Value difference, next month - previous month
    :args: df (pd.DataFrame): all macro data
    :returns:
        _type_: dataframe with binary labels
    """
    result = []   
    for i in range(len(df)-1):
        if (df.iloc[i+1]['Adj Close'] - df.iloc[i]['Adj Close']) < constants.THRESHOLD_CHANGE:
            result.append(0)  # down
        else:
            result.append(1)  # up

    df.drop(df.tail(1).index, inplace=True)  # drop the last line
    df['label'] = result
    return df

results = binary_labels_creation(data)

results.to_csv("macro_data_binary_labels.csv")
print("Binary label creation successfully completed")
