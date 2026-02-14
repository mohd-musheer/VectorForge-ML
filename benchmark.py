import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression


print("Loading dataset...")

df = pd.read_csv("dataset/cars.csv")

# enlarge dataset
df = pd.concat([df] * 10 , ignore_index=True)

print("Rows:", len(df))


y = df["msrp"]
X = df.drop("msrp", axis=1)


start = time.time()

# categorical detection
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(X[cat_cols])

# combine
import numpy as np
X_all = np.hstack([X[num_cols].values, encoded])

# remove constant columns
X_all = X_all[:, X_all.var(axis=0) != 0]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42
)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# train
model = LinearRegression()
model.fit(X_train, y_train)

end = time.time()

print("\nsklearn Time:", end-start,"seconds")
