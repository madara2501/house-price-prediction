import pandas as pd

df = pd.read_csv("india_housing_prices.csv")

small_df = df[
    ["State", "City", "Property_Type"]
].drop_duplicates()

small_df.to_csv("dropdown_data.csv", index=False)

print("Created dropdown_data.csv")