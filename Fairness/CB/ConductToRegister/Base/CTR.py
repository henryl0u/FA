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
from catboost import CatBoostClassifier
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

base_path = "./Fairness/CB/ConductToRegister/Base"

# Save the default standard output
default_stdout = sys.stdout

# Open the file you want to redirect output to
f = open(f"{base_path}/evaluation/prints.txt", "w")

# Change standard output to point to the file
sys.stdout = f

# Load data
training_data = pd.read_excel(
    "./Data/FA/FA Training Data 58-65, Conduct To Register.xlsx"
)
pred66_data = pd.read_excel("./Data/FA/FA Test Data 66, Conduct To Register.xlsx")
pred67_data = pd.read_excel("./Data/FA/FA Test Data 67, Conduct To Register.xlsx")

# Fix only the country column by replacing NaN with "NA"
for df in [training_data, pred66_data, pred67_data]:
    df["country"] = df["country"].fillna("NA")


def replace_nulls_with_nearest_tuition(target_data, source_data):
    for index, row in target_data.iterrows():
        if pd.isnull(row["tuition"]):
            # Filter out rows in source_data where 'tuition' is NA
            valid_tuitions = source_data["tuition"].dropna()
            if not valid_tuitions.empty:
                # Calculate the absolute difference between 'tuition_predicted' and valid tuitions
                nearest_index = (
                    (valid_tuitions - row["tuition_predicted"]).abs().idxmin()
                )
                # Get the nearest tuition from source_data
                nearest_tuition = source_data.at[nearest_index, "tuition"]
                # Replace the null in target_data with the nearest tuition from source_data
                target_data.at[index, "tuition"] = nearest_tuition


# Apply the function to each DataFrame, using training_data as the source for replacements
replace_nulls_with_nearest_tuition(training_data, training_data)
replace_nulls_with_nearest_tuition(pred66_data, training_data)
replace_nulls_with_nearest_tuition(pred67_data, training_data)


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
    "university_caliber": {
        "Top 20": 4,
        "Top 100": 3,
        "Top 5 percent": 2,
        "Beyond": 1,
    },
    "course_completion_status": {
        "Before Interview": 2,
        "After Interview": 2,
        "No Completion": 1,
    },
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
pred66_data = apply_mappings(pred66_data, mappings)
pred67_data = apply_mappings(pred67_data, mappings)


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
        plt.savefig(f"{save_path}/evaluation/confusion_matrix.png")
    plt.show()


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
        file_path = f"{save_path}/evaluation/feature_importance.png"
        plt.savefig(file_path)

    plt.show()


def plot_roc_curve(y_true, y_scores, save_path=None):
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Compute ROC AUC score
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    if save_path:
        plt.savefig(f"{save_path}/evaluation/roc_curve.png")
    plt.show()

    return roc_auc


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

    plt.title("Distribution of Predicted Likelihoods by Actual Interview Conducting")
    plt.xlabel("Predicted Likelihood to Conduct")
    plt.ylabel("Count")
    # Set custom legend labels
    plt.legend(
        title="Actual Interview Conducting", labels=["Conducted", "Not Conducted"]
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
    plt.title("Box Plot of Predicted Likelihoods by Actual Interview Conducting")
    plt.xlabel("Actual Interview Conducting")
    plt.ylabel("Predicted Likelihood")
    plt.xticks([0, 1], ["Not Conducted", "Conducted"])
    # Remove legends from box plot
    if box_plot.legend_ is not None:
        box_plot.legend_.remove()

    if save_path:
        plt.savefig(f"{save_path}/boxplot.png")

    plt.show()


def plot_log_loss_evaluation(model, save_path=None):
    results = model.get_evals_result()

    # Retrieve the keys dynamically based on provided evaluation set names
    train_logloss = results["learn"]["Logloss"]
    val_logloss = results["validation"]["Logloss"]

    # Find the best iteration based on minimum validation log loss
    best_iteration = np.argmin(val_logloss)
    best_val_logloss = val_logloss[best_iteration]

    plt.figure(figsize=(10, 7))
    plt.plot(train_logloss, label="Training loss (Log Loss)", color="blue")
    plt.plot(val_logloss, label="Validation loss (Log Loss)", color="orange")

    # Add a vertical line for the optimal number of iterations
    plt.axvline(
        x=best_iteration,
        color="red",
        linestyle="--",
        label=f"Optimal Iteration: {best_iteration}",
    )

    plt.scatter(
        best_iteration, best_val_logloss, color="red"
    )  # Highlight the optimal point

    plt.xlabel("Number of iterations")
    plt.ylabel("Log Loss")
    plt.title("Log Loss Evaluation for Training and Validation")
    plt.legend()

    if save_path:
        plt.savefig(f"{save_path}/evaluation/evals.png")
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
        plt.savefig(f"{save_path}/evaluation/reliability_curve.png")
    plt.show()


def save_model(model, base_path):
    joblib.dump(model, f"{base_path}/models/model_clf.pkl")


def classification_model(
    data, features, target, param_grid, base_path, calibrate=False
):
    X = data[features]
    y = data[target]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    categorical_features = [
        "role",
        "most_recent_role",
        "industry",
        "hdyhau",
        "prior_education",
        "reason_for_applying",
        "gender",
        "ethnicity",
        "country",
    ]

    grid_search_cv = GridSearchCV(
        estimator=CatBoostClassifier(
            random_seed=42,
            early_stopping_rounds=50,
            silent=True,
            allow_writing_files=False,
            auto_class_weights="Balanced",
            cat_features=categorical_features,
        ),
        param_grid=param_grid,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42),
        n_jobs=-1,
        scoring="f1",
    )

    grid_search_cv.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    best_clf = grid_search_cv.best_estimator_

    if calibrate:
        calibrated_clf = CalibratedClassifierCV(best_clf, method="sigmoid", cv="prefit")
        calibrated_clf.fit(X_train, y_train)
    else:
        calibrated_clf = best_clf

    y_val_pred = calibrated_clf.predict(X_val)
    y_val_probs = calibrated_clf.predict_proba(X_val)[:, 1]

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

    plot_confusion_matrix(y_val, y_val_pred, calibrated_clf, save_path=base_path)
    plot_feature_importance(best_clf, X_train, save_path=base_path)
    plot_log_loss_evaluation(best_clf, save_path=base_path)
    plot_roc_curve(y_val, y_val_probs, save_path=base_path)
    plot_reliability_curve(y_val, y_val_probs, save_path=base_path)

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

    return calibrated_clf


def likelihood_prediction(calibrated_clf, pred_data, pred_name, features, base_path):
    X_test = pred_data[features]

    likelihood_calibrated = calibrated_clf.predict_proba(X_test)[:, 1]

    likelihood_df = pred_data.copy()
    likelihood_df["enrollment_likelihood"] = likelihood_calibrated

    save_dir = f"{base_path}/prediction/{pred_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Save the likelihood predictions to an Excel file for later use
    likelihood_df.to_excel(
        f"{save_dir}/enrollment_likelihood_{pred_name}.xlsx", index=False
    )

    plot_likelihood_distribution(
        pred_data["accepted_offer"], likelihood_calibrated, save_path=save_dir
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
    "university_caliber",
    "prior_education",
    "reason_for_applying",
    "character_count",
    "course_completion_status",
    "tuition",
    "management_leadership_experience",
    "tuition_benefits",
    "english_proficient",
]

target_column = "accepted_offer"

param_grid = {
    "iterations": [500],
}

# Main code for classification model
print("Classification Model:")
calibrated_clf = classification_model(
    training_data,
    feature_columns,
    target_column,
    param_grid,
    base_path,
    calibrate=False,
)

save_model(calibrated_clf, base_path)


# Predict enrollment likelihood for multiple datasets
prediction_datasets = {
    "66": pred66_data,
    "67": pred67_data,
}

for pred_name, pred_data in prediction_datasets.items():
    likelihood_prediction(
        calibrated_clf, pred_data, pred_name, feature_columns, base_path
    )

# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()
