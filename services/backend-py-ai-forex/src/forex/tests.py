import pandas as pd
import constants

reader = pd.read_csv("data/ML_Data_29un2022-train.csv", delimiter=";")
for row in reader["updated_date"]:
    row.split("/")
    row[2] + "-" + str(row[0]).zfill(2)
    values = [
                reader["updated_date"][row],
                row["Regime(redefined)"].lower(),
                float(row["growth_raw_value"].replace(",", "")),
                float(row["inflation_raw_value"].replace(",", "")),
            ]

# print(values)


reader = pd.read_csv("data/EA/ea-data-dev.csv", delimiter=";")
print(reader)



