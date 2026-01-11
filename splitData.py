from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

train, test = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Diabetes_012"],
    random_state=42
)

train.to_csv("data/offline.csv", index=False)
test.to_csv("data/online.csv", index=False)