import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('inst/dataset/heart_disease_b.csv')

print(f"Rows: {len(df)}")

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

start = time.time()
pipe.fit(X_train, y_train)
end = time.time()

pred = pipe.predict(X_test)

print(f"\nTrain Time: {end - start} sec")
print(f"Accuracy: {accuracy_score(y_test, pred)}")

# Example new patient data using the feature structure of the dataset
new_patient = X_test.iloc[[0]]

result = pipe.predict(new_patient)
print(f"\n Prediction for Breast Cancer (0=Malignant, 1=Benign): {result[0]}")