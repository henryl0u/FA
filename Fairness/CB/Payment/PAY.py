import sys
import pandas as pd
import joblib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math


base_path = "./Fairness/CB/Payment"

# Save the default standard output
default_stdout = sys.stdout

# Open the file you want to redirect output to
f = open(f"{base_path}/evaluation/prints.txt", "w")

# Change standard output to point to the file
sys.stdout = f

# Load data
training_data = pd.read_excel("./Data/FA Training Data 50-59, Tuition.xlsx")
pred60_data = pd.read_excel("./Data/FA Test Data 60, Tuition.xlsx")

# Fix only the country column by replacing NaN with "NA"
for df in [training_data, pred60_data]:
    df["country"] = df["country"].fillna("NA")

for df in [training_data, pred60_data]:
    df = df[df["payment_amount"] > 0]


# Define values for mapping
mappings = {
    "company_caliber": {"Average": 1, "Self-Employed": 1, "Good": 2, "Elite": 3},
    "years_full_time_experience": {
        "0_1": 1,
        "2_6": 1,
        "3_6": 1,
        "7_10": 2,
        "11_15": 3,
        "16_plus": 4,
    },
    # "academic_institution_caliber": {
    #     "Not highly regarded": 1,
    #     "Average": 2,
    #     "Good": 3,
    #     "Elite": 4,
    # },
    "management_leadership_experience": {
        "Yes": 2,
        "No": 1,
    },
    "tuition_benefits": {
        "Eligible": 2,
        "Unknown/Not eligible": 1,
    },
    "english_proficient": {
        True: 2,
        False: 1,
    },
}


# Apply mapping to columns
def apply_mappings(data, mappings):
    for column, mapping in mappings.items():
        data[column] = data[column].map(mapping)
    return data


training_data = apply_mappings(training_data, mappings)
pred60_data = apply_mappings(pred60_data, mappings)


def plot_feature_importance(clf, X_train, save_path=None):

    # Compute feature importance
    feature_importance = pd.Series(
        clf.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.title("Feature Importance")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()

    # Determine correct save_path
    if save_path:
        file_path = f"{save_path}/feature_importance.png"
        plt.savefig(file_path)

    plt.show()


def plot_regression_diagnostics(actual, predicted, pred_name, training_data, save_path):

    # Round predicted values to the nearest tuition value in training data
    unique_tuition_values = training_data.unique()
    rounded_payment = pd.Series(predicted).apply(
        lambda x: min(unique_tuition_values, key=lambda y: abs(y - x))
    )

    # Calculate residuals in absolute values
    residuals = actual - rounded_payment

    if pred_name:
        print(f"\nResiduals for {pred_name}:")
    print("Sum of predicted payment:", np.sum(rounded_payment))
    print("Sum of actual payment:", np.sum(actual))
    print("Sum of residuals:", residuals.sum())

    # Create a DataFrame with actual outcomes, predicted values, and residuals
    results_df = pd.DataFrame(
        {
            "Actual": actual,
            "Predicted": rounded_payment,
            "Residuals": residuals,
        }
    )

    # Plot histogram of residuals
    plt.figure(figsize=(12, 6))
    sns.histplot(results_df["Residuals"], bins=20, color="purple")
    plt.title("Distribution of Residuals")
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Count")

    if save_path:
        plt.savefig(f"{save_path}/residuals_histogram_{pred_name}.png")
    plt.show()

    # Plot actual vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(
        actual, rounded_payment, color="blue", alpha=0.6
    )
    plt.plot(
        [min(actual), max(actual)],
        [min(actual), max(actual)],
        color="red",
        lw=2,
    )
    plt.title("Actual vs Predicted Payment")
    plt.xlabel("Actual Payment")
    plt.ylabel("Predicted Payment")
    plt.grid(True)

    if save_path:
        plt.savefig(f"{save_path}/actual_vs_predicted_{pred_name}.png")
    plt.show()

    # Plot boxplot of residuals
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=residuals, color="green")
    plt.title("Box Plot of Residuals")
    plt.ylabel("Residuals (Actual - Predicted)")

    if save_path:
        plt.savefig(f"{save_path}/residuals_boxplot_{pred_name}.png")
    plt.show()


def save_model(model, base_path):
    joblib.dump(model, f"{base_path}/model/model.pkl")


def regression_model(data, features, target, param_grid, base_path):
    X = data[features]
    y = data[target]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = [
        "role",
        "most_recent_role",
        "industry",
        "hdyhau",
        "prior_education",
        "reason_for_applying",
        # "course_completion_status",
        "gender",
        "ethnicity",
        "country",
    ]

    grid_search_cv = GridSearchCV(
        estimator=CatBoostRegressor(
            random_seed=42,
            early_stopping_rounds=50,
            silent=True,
            allow_writing_files=False,
            cat_features=categorical_features,
        ),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )

    grid_search_cv.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    best_model = grid_search_cv.best_estimator_

    y_val_pred = best_model.predict(X_val)

    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = math.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)

    print("Best Parameters:", grid_search_cv.best_params_)
    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Validation R² Score: {r2:.4f}")

    save_dir = f"{base_path}/evaluation/"
    os.makedirs(save_dir, exist_ok=True)

    plot_feature_importance(best_model, X_train, save_path=save_dir)

    save_dir = f"{base_path}/prediction/val"
    os.makedirs(save_dir, exist_ok=True)
    plot_regression_diagnostics(y_val, y_val_pred, "Validation", y, save_path=save_dir)

    return best_model


def tuition_prediction(model, pred_data, pred_name, training_data, features, base_path):
    X_test = pred_data[features]
    predicted_tuition = model.predict(X_test)

    save_dir = f"{base_path}/prediction/{pred_name}"
    os.makedirs(save_dir, exist_ok=True)

    plot_regression_diagnostics(pred_data["payment_amount"], predicted_tuition, pred_name, training_data, save_dir)


feature_columns = [
    "age",
    "company_caliber",
    "years_full_time_experience",
    "role",
    "most_recent_role",
    "industry",
    "gender",
    "ethnicity",
    "country",
    "hdyhau",
    "salary_range",
    "prior_education",
    "reason_for_applying",
    "character_count",
    "management_leadership_experience",
    "tuition_benefits",
    "english_proficient",
]

target_column = "payment_amount"
param_grid = {}

print("Regression Model:")
model = regression_model(
    training_data, feature_columns, target_column, param_grid, base_path
)

save_model(model, base_path)

tuition_prediction(model, pred60_data, "60", training_data["payment_amount"], feature_columns, base_path)



# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()
