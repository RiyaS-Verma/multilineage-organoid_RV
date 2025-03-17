import pandas as pd

df = pd.read_csv("5280_stretched_chamber2_1.5Hz.csv")

df = df[["time","ROI_7","ROI_8"]]
df.to_csv("debug_2_cols.csv", index=False)