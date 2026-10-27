import pandas as pd

X = pd.read_csv(r'./secom/secom.data', sep=r"\s+", header=None)
labels = pd.read_csv(
    r'./secom/secom_labels.data',
    sep=r"\s+",
    header=None
)

X.columns = [f"Sensor_{i+1}" for i in range(X.shape[1])]
X.insert(0, "Timestamp", labels.iloc[:, 1])
X["Pass_Fail"] = labels.iloc[:, 0].map({-1: "Pass", 1: "Fail"})

X.to_csv("secom_combined.csv", index=False)
X.to_excel("secom_combined.xlsx", index=False)

print("Shape:", X.shape)
print(X["Pass_Fail"].value_counts())