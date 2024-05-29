import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %%
file_path = Path(os.getcwd()) / "data/ordinal_classification_sample.xlsx"
df = pd.read_excel(file_path.resolve())
df.head()

# %%
df.dtypes
# %%
print((df.isna().sum() / df.shape[0] * 100).round())
df = df.drop(columns="D")
# %%
X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()


# %%
# Transformation
def ordinal_transformation(target: np.array) -> np.array:
    vals = np.sort(np.unique(target))
    n = len(vals)  # => n-1 columns
    m = len(target)
    ret = np.empty((m, n - 1), dtype=int)
    for idx, v in enumerate(vals[:-1]):
        col = (target > v).astype(int)
        ret[:, idx] = col
    return ret, vals.tolist()


def inverse_ordinal_transformation(transformed: np.array, values):
    m, n = transformed.shape
    assert n + 1 == len(values)
    dt = type(values[0])
    ret = np.empty(m, dtype=dt)
    tmp = transformed.T
    col0 = tmp[0]
    ret[np.where(col0 == 0)] = values[0]
    for idx, col in enumerate(tmp[1:]):
        prev_col = tmp[idx]
        ret[(prev_col == 1) & (col == 0)] = values[idx + 1]
    col_n = tmp[-1]
    ret[col_n == 1] = values[n]
    return ret


# %%
y_transformed, values = ordinal_transformation(y)

# %%
X_train, X_test, y_train, y_test, y_transformed_train, y_transformed_test = train_test_split(
    X, y, y_transformed, test_size=0.2, random_state=42)

# %%
# Logistic regression with original values
# Set up pipeline
steps = [("scaler", StandardScaler()),
         ("logreg", LogisticRegression())]
pipeline = Pipeline(steps)

# Set up GridSearchCV
params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],
          "logreg__C": np.linspace(0.001, 1.0, 10)}

tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)

y_pred = tuning.predict(X_test)

# Compute and print performance
print(f"Tuned Logistic Regression Parameters: {tuning.best_params_},\n"
      f"Training accuracy: {tuning.score(X_train, y_train)}, \n"
      f"Test accuracy: {tuning.score(X_test, y_test)}")


# %%
# Logistic regression with transformation

def my_fit(feature_matrix, ordinal_target):
    models = []
    for col in ordinal_target.T:
        search = GridSearchCV(pipeline, param_grid=params)
        search.fit(feature_matrix, col)
        models.append(search.best_estimator_)
    return models


def my_predict_proba(feature_matrix, models):
    m = feature_matrix.shape[0]
    n = len(models) + 1  # n-1 models
    probs_k = np.empty((m, n))
    probs_k[:, 0] = models[0].predict_proba(feature_matrix)[:, 0]  # 0 == no
    assert models[0].classes_[0] == 0
    assert models[0].classes_[1] == 1
    for idx, mdl in enumerate(models[1:]):
        prev_mdl = models[idx]
        assert mdl.classes_[0] == 0
        assert mdl.classes_[1] == 1
        probs_k[:, idx + 1] = np.clip((prev_mdl.predict_proba(feature_matrix)[:, 1]
                                       - mdl.predict_proba(feature_matrix)[:, 1]),
                                      a_min=0.0, a_max=None)
    probs_k[:, n - 1] = models[-1].predict_proba(feature_matrix)[:, 1]
    return probs_k


def my_predict(feature_matrix, models, values):
    probs_k = my_predict_proba(feature_matrix, models)
    idx = np.argmax(probs_k, axis=1)
    return np.array([values[i] for i in idx])


# %%
mdls = my_fit(X_train, y_transformed_train)
y_train_preds = my_predict(X_train, mdls, values)
y_test_preds = my_predict(X_test, mdls, values)

# %%
print(f"Training accuracy: {accuracy_score(y_train, y_train_preds)}, \n"
      f"Test accuracy: {accuracy_score(y_test, y_test_preds)}")


