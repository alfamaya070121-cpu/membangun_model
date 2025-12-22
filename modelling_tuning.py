import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# ======================
# CONNECT TO DAGSHUB
# ======================
dagshub.init(
    repo_owner="alfamaya070121-cpu",
    repo_name="Modelling",
    mlflow=True
)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(
    "/content/preprocessing/namadataset_preprocessing/ibm_hr_attrition_preprocessed.csv"
)

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# HYPERPARAMETER TUNING
# ======================
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear"]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    scoring="f1",
    cv=5
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# ======================
# EVALUATION
# ======================
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

# ======================
# MLFLOW MANUAL LOGGING
# ======================
with mlflow.start_run():

    mlflow.log_params(grid.best_params_)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    mlflow.sklearn.log_model(best_model, "model")

    # -------- ARTIFACT 1: Confusion Matrix --------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    # -------- ARTIFACT 2: Feature Importance --------
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": best_model.coef_[0]
    })

    coef_df.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

print("Training selesai, data terkirim ke DagsHub")
