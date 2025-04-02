import sys
import pandas as pd
import joblib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    MinMaxScaler,
)
from sklearn.impute import SimpleImputer

base_path = "./Fairness/LR/base/OfferToRegister/"

# Save the default standard output
default_stdout = sys.stdout

# Open the file you want to redirect output to
f = open(f"{base_path}/evaluation/prints.txt", "w")

# Change standard output to point to the file
sys.stdout = f

# Load data
training_data = pd.read_excel("./Data/FA Training Data 50-59, Offer To Register.xlsx")
pred60_data = pd.read_excel("./Data/FA Test Data 60, Offer To Register.xlsx")

# Fix only the country column by replacing NaN with "NA"
for df in [training_data, pred60_data]:
    df["country"] = df["country"].fillna("NA")


def plot_confusion_matrix(y_true, y_pred, clf, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=clf.classes_)

    # Replace the default labels (0, 1) with "False" and "True"
    display_labels = ["False", "True"]

    # Create confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.title("Confusion Matrix")

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.show()


def plot_feature_importance(model, features, save_path=None):
    # Get the feature importance from the model's coefficients
    if hasattr(model, "coef_"):
        feature_importance = model.coef_[0]  # For Logistic Regression, it's a 1D array
    else:
        print("Model does not have coef_ attribute for feature importance.")
        return

    # Create a DataFrame for easy plotting
    importance_df = pd.DataFrame(
        {"Feature": features, "Importance": feature_importance}
    )

    # Sort the importance values
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Plot the feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    if save_path:
        plt.savefig(f"{save_path}/feature_importance.png")
    plt.show()


def plot_roc_curve(y_true, y_scores, save_path=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Calculate the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.scatter(
        optimal_fpr,
        optimal_tpr,
        color="red",
        marker="o",
        label=f"Optimal Threshold: {optimal_threshold:.2f}",
    )
    plt.text(
        optimal_fpr,
        optimal_tpr,
        f"Threshold: {optimal_threshold:.2f}",
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()

    if save_path:
        plt.savefig(f"{save_path}/roc_curve.png")
    plt.show()

    return roc_auc, optimal_threshold


def plot_likelihood_distribution(y_true, y_probs, save_path=None):

    # Create a DataFrame with actual outcomes and predicted probabilities
    results_df = pd.DataFrame({"Actual": y_true, "Predicted Likelihood": y_probs})

    # Plot histogram of predicted likelihoods segmented by actual class
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=results_df,
        x="Predicted Likelihood",
        hue="Actual",
        bins=20,
        palette="viridis",
        stat="count",
        common_norm=False,
    )

    plt.title("Distribution of Predicted Likelihoods by Actual Enrollment")
    plt.xlabel("Predicted Likelihood to Enroll")
    plt.ylabel("Count")
    # Set custom legend labels
    plt.legend(
        title="Actual Enrollment",
        labels=["Enrolled", "Not Enrolled"],
    )

    if save_path:
        plt.savefig(f"{save_path}/histogram.png")
    plt.show()

    # Plot box plots for a clearer view of distribution
    plt.figure(figsize=(8, 6))
    box_plot = sns.boxplot(
        x="Actual",
        y="Predicted Likelihood",
        data=results_df,
        hue="Actual",  # Assign hue parameter
        palette="viridis",
        dodge=False,  # Ensure one box per category
    )
    plt.title("Box Plot of Predicted Likelihoods by Actual Enrollment")
    plt.xlabel("Actual Enrollment")
    plt.ylabel("Predicted Likelihood")
    plt.xticks([0, 1], ["Not Enrollment", "Enrollment"])
    # Remove legends from box plot
    if box_plot.legend_ is not None:
        box_plot.legend_.remove()

    if save_path:
        plt.savefig(f"{save_path}/boxplot.png")

    plt.show()


def plot_reliability_curve(y_true, y_probs, save_path=None):
    # Calculate the calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)

    # Plot the reliability curve
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Reliability Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Curve")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(f"{save_path}/reliability_curve.png")
    plt.show()


def best_threshold_combined(y_true, y_probs, alpha=0.3, save_path=None):
    thresholds = np.arange(
        0, 1.01, 0.05
    )  # Define thresholds from 0 to 1 with a step of 0.05
    combined_scores = []

    # Calculate combined score for each threshold
    for threshold in thresholds:
        predictions = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, predictions)
        bal_acc = balanced_accuracy_score(y_true, predictions)

        # Weighted average of F1 and Balanced Accuracy
        combined_score = alpha * f1 + (1 - alpha) * bal_acc
        combined_scores.append(combined_score)

    # Find the optimal threshold
    best_index = np.argmax(combined_scores)
    best_threshold = thresholds[best_index]
    max_combined_score = combined_scores[best_index]

    f1_scores = []
    balanced_accuracies = []

    for threshold in thresholds:
        predictions = (y_probs >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, predictions))
        balanced_accuracies.append(balanced_accuracy_score(y_true, predictions))

    # Plotting the two metrics
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label="F1 Score", color="blue", marker="o")
    plt.plot(
        thresholds,
        balanced_accuracies,
        label="Balanced Accuracy",
        color="orange",
        marker="o",
    )

    # Adding threshold line
    plt.title("F1 Score and Balanced Accuracy by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.axvline(
        best_threshold,
        linestyle="--",
        color="red",
        label=f"Best Threshold: {best_threshold:.2f}",
    )

    # Annotate the optimal threshold value
    plt.text(
        best_threshold,
        max_combined_score,
        f"{best_threshold:.2f}",
        color="red",
        fontsize=12,
        verticalalignment="bottom",
    )

    # Set x-ticks for finer resolution
    plt.xticks(np.arange(0, 1.05, 0.05))  # X-axis ticks every 0.05
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(f"{save_path}/threshold_evaluation.png")
    plt.show()

    return best_threshold, max_combined_score


def save_model(model, base_path):
    joblib.dump(model, f"{base_path}/model/model.pkl")


def classification_model(
    data, features, target, param_grid, base_path, seed=42, calibrate=False
):
    categorical_features = [
        "role",
        "most_recent_role",
        "industry",
        "hdyhau",
        "prior_education",
        "reason_for_applying",
        "company_caliber",
        "gender",
        "ethnicity",
        "country",
        "english_proficient",
        "tuition_benefits",
        "management_leadership_experience",
        "years_full_time_experience",
    ]

    numerical_features = ["age", "salary_range", "character_count"]

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),  # Handle missing values
            ("onehot", OneHotEncoder(handle_unknown="ignore")),  # One-Hot Encoding
        ]
    )

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),  # Handle missing values
            ("scaler", MinMaxScaler()),  # Scaling the numerical features
        ]
    )

    # Combine both preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Create a pipeline that first preprocesses the data and then fits the model
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=42,
                ),
            ),
        ]
    )

    X = data[features]
    y = data[target]

    # Initial Train-Validation Split
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Split Training Data Again for Calibration (80% Train, 20% Calibration)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=seed
    )

    # Grid Search with Logistic Regression
    grid_search_cv = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1",
    )

    grid_search_cv.fit(X_train, y_train)

    best_clf = grid_search_cv.best_estimator_

    if calibrate:
        calibrated_clf = CalibratedClassifierCV(best_clf, method="sigmoid", cv="prefit")
        calibrated_clf.fit(X_calib, y_calib)
    else:
        calibrated_clf = best_clf

    save_dir = f"{base_path}/evaluation/"
    os.makedirs(save_dir, exist_ok=True)

    y_val_probs = calibrated_clf.predict_proba(X_val)[:, 1]
    best_threshold, max_combined_score = best_threshold_combined(
        y_val, y_val_probs, save_path=save_dir
    )

    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Maximum Combined Score: {max_combined_score:.4f}")

    y_val_pred = (y_val_probs >= best_threshold).astype(int)
    # y_val_pred = calibrated_clf.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
    val_f1_score = f1_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_probs)

    print("Best Parameters:")
    for param, value in grid_search_cv.best_params_.items():
        print(f"{param}: {value}")

    print(f"\nValidation Set Accuracy: {val_accuracy:.4f}")
    print(f"Validation Set Balanced Accuracy: {val_balanced_accuracy:.4f}")
    print(f"Validation Set F1 Score: {val_f1_score:.4f}")
    print(f"Validation Set ROC AUC Score: {val_roc_auc:.4f}\n")
    print(classification_report(y_val, y_val_pred))

    plot_confusion_matrix(y_val, y_val_pred, calibrated_clf, save_path=save_dir)
    plot_feature_importance(best_clf, X_train, save_path=save_dir)
    plot_roc_curve(y_val, y_val_probs, save_path=save_dir)
    plot_reliability_curve(y_val, y_val_probs, save_path=save_dir)

    # Define the save directory path based on prediction name
    save_dir = f"{base_path}/prediction/val"

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Collect the results in a DataFrame for better visualization
    likelihood_df = X_val.copy()  # Keep all original features
    likelihood_df["enrollment_likelihood"] = y_val_probs

    # Save the likelihood predictions to an Excel file for later use
    likelihood_df.to_excel(
        f"{save_dir}/enrollment_likelihood_val.xlsx",
        index=False,
    )

    # Plot the distribution of predicted likelihoods
    plot_likelihood_distribution(y_val, y_val_probs, save_path=save_dir)

    return calibrated_clf, best_threshold


def likelihood_prediction(
    calibrated_clf, pred_data, pred_name, features, best_threshold, base_path
):
    X_test = pred_data[features]

    likelihood_calibrated = calibrated_clf.predict_proba(X_test)[:, 1]
    # predictions = calibrated_clf.predict(X_test)
    predictions = (likelihood_calibrated >= best_threshold).astype(int)

    likelihood_df = pred_data.copy()
    likelihood_df["enrollment_likelihood"] = likelihood_calibrated

    save_dir = f"{base_path}/prediction/{pred_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Save the likelihood predictions to an Excel file for later use
    likelihood_df.to_excel(
        f"{save_dir}/enrollment_likelihood_{pred_name}.xlsx", index=False
    )

    plot_confusion_matrix(
        pred_data["registered"], predictions, calibrated_clf, save_path=save_dir
    )

    plot_likelihood_distribution(
        pred_data["registered"], likelihood_calibrated, save_path=save_dir
    )


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
    "payment_amount",
    "management_leadership_experience",
    "tuition_benefits",
    "english_proficient",
]

target_column = "registered"

# param_grid = {
#     'classifier__C': [0.01, 0.1, 1, 10],  # Inverse of regularization strength (default: 1)
#     'classifier__max_iter': [1000],  # Max number of iterations for convergence (default: 100)
#     'classifier__class_weight': ['balanced', None],  # Class weights for imbalance handling (default: None)
# }

param_grid = {
    'classifier__C': [1],  # Inverse of regularization strength (default: 1)
    'classifier__max_iter': [1000],  # Max number of iterations for convergence (default: 100)
    'classifier__class_weight': ['balanced'],  # Class weights for imbalance handling (default: None)
}

# Main code for classification model
print("Classification Model:")
calibrated_clf, best_threshold = classification_model(
    training_data,
    feature_columns,
    target_column,
    param_grid,
    base_path,
    seed=9,
    calibrate=True,
)

save_model(calibrated_clf, base_path)

# Predict enrollment likelihood for multiple datasets
prediction_datasets = {
    "60": pred60_data,
}

for pred_name, pred_data in prediction_datasets.items():
    likelihood_prediction(
        calibrated_clf, pred_data, pred_name, feature_columns, best_threshold, base_path
    )

# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()
