import sys
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os
from sklearn.svm import SVC
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
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds


base_path = "./Fairness/SVM/egr/InviteToConduct/"

# Save the default standard output
default_stdout = sys.stdout

# Open the file you want to redirect output to
f = open(f"{base_path}/evaluation/prints.txt", "w")

# Change standard output to point to the file
sys.stdout = f

# Load data
training_data = pd.read_excel("./Data/FA Training Data 50-59, Invite To Conduct.xlsx")
pred60_data = pd.read_excel("./Data/FA Test Data 60, Invite To Conduct.xlsx")

# Fix only the country column by replacing NaN with "NA"
for df in [training_data, pred60_data]:
    df["country"] = df["country"].fillna("NA")


def plot_confusion_matrix(y_true, y_pred, clf, save_path=None):
    base_clf = clf.predictors_[0] if hasattr(clf, "predictors_") else clf
    cm = confusion_matrix(y_true, y_pred, labels=base_clf.classes_)

    display_labels = ["False", "True"]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(f"{save_path}/confusion_matrix.png")
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
        plt.savefig(f"{save_path}/evals.png")
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
    thresholds = np.arange(0, 1.01, 0.05)  # Define thresholds from 0 to 1 with a step of 0.05
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
    plt.plot(thresholds, f1_scores, label="F1 Score", color="blue", marker='o')
    plt.plot(thresholds, balanced_accuracies, label="Balanced Accuracy", color="orange", marker='o')
    
    # Adding threshold line
    plt.title("F1 Score and Balanced Accuracy by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.axvline(best_threshold, linestyle="--", color="red", label=f"Best Threshold: {best_threshold:.2f}")
    
    # Annotate the optimal threshold value
    plt.text(best_threshold, max_combined_score, f'{best_threshold:.2f}', color='red', fontsize=12, verticalalignment='bottom')

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

def egr_predict_proba(egr_model, X):
    probas = np.zeros(len(X))
    for weight, predictor in zip(egr_model.weights_, egr_model.predictors_):
        probas += weight * predictor.predict_proba(X)[:, 1]
    return probas
    
def classification_model(
    data, features, target, param_grid, base_path, seed=42,
    fairness_metric="equal_opportunity"
):
    categorical_features = [
        "role", "most_recent_role", "industry", "hdyhau", "prior_education",
        "reason_for_applying", "company_caliber", "gender", "ethnicity", "country",
        "english_proficient", "tuition_benefits", "management_leadership_experience",
        "years_full_time_experience",
    ]
    numerical_features = ["age", "salary_range", "character_count"]

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)
    ])

    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    X = data[features]
    y = data[target]
    A = data["ethnicity"]  # fixed sensitive attribute

    X_train_full, X_val, y_train_full, y_val, A_train_full, A_val = train_test_split(
        X, y, A, test_size=0.2, stratify=y, random_state=seed
    )

    X_train, X_calib, y_train, y_calib, A_train, A_calib = train_test_split(
        X_train_full, y_train_full, A_train_full, test_size=0.2,
        stratify=y_train_full, random_state=seed
    )

    # Apply preprocessing
    fair_pipeline = Pipeline([("preprocessor", preprocessor)])
    X_train_proc = fair_pipeline.fit_transform(X_train)
    X_val_proc = fair_pipeline.transform(X_val)

    # Select fairness constraint
    if fairness_metric == "equal_opportunity":
        constraint = EqualizedOdds()
    elif fairness_metric == "demographic_parity":
        constraint = DemographicParity()
    else:
        raise ValueError("Unsupported fairness metric.")

    base_model = SVC(
        C = param_grid['classifier__C'][0],
        kernel=param_grid['classifier__kernel'][0],
        gamma=param_grid['classifier__gamma'][0],
        class_weight=param_grid['classifier__class_weight'][0],
        probability=True,
        random_state=seed,
    )

    print(f"Training EGR model using fairness constraint: {fairness_metric} and sensitive attribute: ethnicity")
    egr_model = ExponentiatedGradient(base_model, constraints=constraint)
    egr_model.fit(X_train_proc, y_train, sensitive_features=A_train)

    y_val_probs = egr_predict_proba(egr_model, X_val_proc)

    save_dir = f"{base_path}/evaluation/"
    os.makedirs(save_dir, exist_ok=True)

    best_threshold, max_combined_score = best_threshold_combined(
        y_val, y_val_probs, save_path=save_dir
    )

    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Maximum Combined Score: {max_combined_score:.4f}")

    y_val_pred = (y_val_probs >= best_threshold).astype(int)

    print(classification_report(y_val, y_val_pred))
    #plot_confusion_matrix(y_val, y_val_pred, egr_model, save_path=save_dir)
    #plot_roc_curve(y_val, y_val_probs, save_path=save_dir)
    #plot_reliability_curve(y_val, y_val_probs, save_path=save_dir)

    save_dir = f"{base_path}/prediction/val"
    os.makedirs(save_dir, exist_ok=True)

    likelihood_df = X_val.copy()
    likelihood_df["interview_likelihood"] = y_val_probs

    likelihood_df.to_excel(
        f"{save_dir}/interview_likelihood_val.xlsx", index=False
    )
    #plot_likelihood_distribution(y_val, y_val_probs, save_path=save_dir)

    return egr_model, best_threshold, fair_pipeline


def likelihood_prediction(calibrated_clf, pred_data, pred_name, features, best_threshold, base_path, preprocessor_pipeline):
    X_pred_raw = pred_data[features]
    X_pred_proc = preprocessor_pipeline.transform(X_pred_raw)

    # Convert to dense if needed
    if hasattr(X_pred_proc, "toarray"):
        X_pred_proc = X_pred_proc.toarray()

    # Get probabilities from EGR predictors
    if hasattr(calibrated_clf, "predictors_"):
        likelihood_calibrated = egr_predict_proba(calibrated_clf, X_pred_proc)
    else:
        likelihood_calibrated = calibrated_clf.predict_proba(X_pred_proc)[:, 1]

    predictions = (likelihood_calibrated >= best_threshold).astype(int)

    likelihood_df = pred_data.copy()
    likelihood_df["interview_likelihood"] = likelihood_calibrated

    save_dir = f"{base_path}/prediction/{pred_name}"
    os.makedirs(save_dir, exist_ok=True)

    likelihood_df.to_excel(
        f"{save_dir}/interview_likelihood_{pred_name}.xlsx", index=False
    )

    # plot_confusion_matrix(
    #     pred_data["did_interview"], predictions, calibrated_clf, save_path=save_dir
    # )

    # plot_likelihood_distribution(
    #     pred_data["did_interview"], likelihood_calibrated, save_path=save_dir
    # )



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

target_column = "did_interview"

# param_grid = {
#     'classifier__C': [0.01, 0.1, 1, 10],             # Regularization (like LR)
#     'classifier__kernel': ['linear', 'rbf'],         # Linear for interpretability, RBF for non-linear
#     'classifier__gamma': ['scale', 'auto'],          # Kernel coefficient (relevant for RBF)
#     'classifier__class_weight': ['balanced', None],  # Handle class imbalance
# }

param_grid = {
    'classifier__C': [0.01],             # Regularization (like LR)
    'classifier__kernel': ['linear'],         # Linear for interpretability, RBF for non-linear
    'classifier__gamma': ['scale'],          # Kernel coefficient (relevant for RBF)
    'classifier__class_weight': [None],  # Handle class imbalance
}


# Main code for classification model
print("Classification Model:")

calibrated_clf, best_threshold, fitted_preprocessor = classification_model(
    training_data,
    feature_columns,
    target_column,
    param_grid,
    base_path,
    seed=0,
    fairness_metric="demographic_parity"  # or "demographic_parity"
)

save_model(calibrated_clf, base_path)
# Save the preprocessor pipeline
joblib.dump(fitted_preprocessor, f"{base_path}/model/preprocessor.pkl")


# Predict enrollment likelihood for multiple datasets
prediction_datasets = {
    "60": pred60_data,
}

# Call with preprocessor
for pred_name, pred_data in prediction_datasets.items():
    likelihood_prediction(
        calibrated_clf,
        pred_data,
        pred_name,
        feature_columns,
        best_threshold,
        base_path,
        preprocessor_pipeline=fitted_preprocessor
    )

# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()
