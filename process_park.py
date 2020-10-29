# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

testsize = 0.2
rstate = 69

df = pd.read_csv("data/raw/train_data.txt", header=None)
# pf = ProfileReport(df, title="Dataset Report", explorative=True)
# pf.to_file("data/reporting/raw_df_pp_report.html")

# print(df)
y = df.iloc[:, 28]
X = df.drop(columns=["0", "22", "27", "28"])
# print(y)
# print(X)

min_max = MinMaxScaler()

X = min_max.fit_transform(X)


pca = PCA()

X = pca.fit_transform(X)[:, 0:6]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=testsize, random_state=rstate
)


np.savetxt("data/cleaned/park_X_train.csv", X_train, delimiter=",")
np.savetxt("data/cleaned/park_y_train.csv", y_train, delimiter=",")
np.savetxt("data/cleaned/park_X_test.csv", X_test, delimiter=",")
np.savetxt("data/cleaned/park_y_test.csv", y_test, delimiter=",")
