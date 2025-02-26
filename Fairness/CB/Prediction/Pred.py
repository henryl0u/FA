import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    auc,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

base_path = "./Fairness/CB/Prediction"

# Save the default standard output
default_stdout = sys.stdout

# Open the file you want to redirect output to
f = open(f"{base_path}/prints.txt", "w")

# Change standard output to point to the file
sys.stdout = f

# Load data
training_data = pd.read_excel("./Data/FA Training Data 50-59, Offer To Register.xlsx")
pred60_data = pd.read_excel("./Data/FA Prediction Data 60.xlsx")

# Fix only the country column by replacing NaN with "NA"
for df in [training_data, pred60_data]:
    df["country"] = df["country"].fillna("NA")

# Handle null tuition values (loss due to cost from advertising)
pred60_data["payment_amount"] = pred60_data["payment_amount"].fillna(
    training_data["base_payment"]
)
pred60_data.rename(columns={"payment_amount": "payment_amount_actual"}, inplace=True)
pred60_data["payment_amount"] = pred60_data["predicted_payment"]


# model_payment = joblib.load("./Fairness/CB/Payment/model/model.pkl")

# features_payment = [
#     "age",
#     "company_caliber",
#     "years_full_time_experience",
#     "role",
#     "most_recent_role",
#     "industry",
#     "gender",
#     "ethnicity",
#     "country",
#     "hdyhau",
#     "salary_range",
#     "prior_education",
#     "reason_for_applying",
#     "character_count",
#     "management_leadership_experience",
#     "tuition_benefits",
#     "english_proficient",
# ]

# def tuition_prediction(model, pred_data, training_data, features):
#     X_test = pred_data[features]
#     predicted_tuition = model.predict(X_test)

#     prediction_df = pred_data.copy()
#     unique_tuition_values = training_data["payment_amount"].unique()

#     # Ensure we iterate over individual predicted values
#     prediction_df["payment_amount"] = [
#         min(unique_tuition_values, key=lambda y: abs(y - pred)) for pred in predicted_tuition
#     ]

#     return prediction_df


# pred60_data = tuition_prediction(model_payment, pred60_data, training_data, features_payment)

# Load your pretrained models and scalers
model_ITC = joblib.load("./Fairness/CB/InviteToConduct/model/model.pkl")

features_ITC = [
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


def conduct_predictions(
    model,
    pred_data,
    features,
):
    X_pred = pred_data[features]

    # Run the model's prediction (classification) on the original data
    predictions = model.predict_proba(X_pred)[:, 1]

    output = pred_data.copy()
    output["interview_likelihood"] = predictions

    return output


pred60_data = conduct_predictions(
    model_ITC,
    pred60_data,
    features_ITC,
)


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
        plt.savefig(f"{save_path}/roc_curve.png")
    plt.show()

    return roc_auc


model_OTR = joblib.load(f"Fairness/CB/OfferToRegister/model/model.pkl")

features_OTR = [
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


def likelihood_prediction(calibrated_clf, pred_data, features):
    X_test = pred_data[features]

    likelihood_calibrated = calibrated_clf.predict_proba(X_test)[:, 1]

    likelihood_df = pred_data.copy()
    likelihood_df["enrollment_likelihood"] = likelihood_calibrated

    return likelihood_df


pred60_data = likelihood_prediction(model_OTR, pred60_data, features_OTR)


def optimal_enrollment_threshold(y_true, y_probs, metric="f1"):
    thresholds = np.arange(0, 1.01, 0.01)
    best_threshold = 0.5
    best_score = 0
    score = 0
    
    for threshold in thresholds:
        predictions = (y_probs >= threshold).astype(int)
    
        if metric == "f1":
            score = f1_score(y_true, predictions)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, predictions)
    
        # Check if we found a better score
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def enrollment_prediction(model, pred_data, pred_name, base_path):
    enrollment_likelihood = pred_data["enrollment_likelihood"]
    interview_likelihood = pred_data["interview_likelihood"]

    # Calculate combined likelihood
    likelihood = enrollment_likelihood * interview_likelihood

    # Finding the optimal threshold
    optimal_threshold, _ = optimal_enrollment_threshold(
        pred_data["registered"], likelihood, metric="f1"
    )

    # Setting predicted enrollment based on the optimal threshold
    # pred_data["predicted_enrollment"] = (likelihood > optimal_threshold).astype(bool)
    pred_data["predicted_enrollment"] = (pred_data["enrollment_likelihood"] > 0.55) & (
        pred_data["interview_likelihood"] > 0.5
    )
    pred_data["predicted_enrollment"] = pred_data["predicted_enrollment"].astype(bool)

    save_dir = f"{base_path}/{pred_name}"
    os.makedirs(save_dir, exist_ok=True)

    plot_confusion_matrix(
        pred_data["registered"],
        pred_data["predicted_enrollment"],
        model,
        save_path=save_dir,
    )
    plot_roc_curve(pred_data["registered"], likelihood, save_path=save_dir)

    pred_data.to_excel(f"{save_dir}/predictions.xlsx", index=False)

    print(f"Predictions for {pred_name}:")
    # Calculate and print classification report
    print("Classification Report:")
    print(
        classification_report(
            pred_data["registered"], pred_data["predicted_enrollment"]
        )
    )

    # Calculate and print accuracy score
    accuracy = accuracy_score(
        pred_data["registered"], pred_data["predicted_enrollment"]
    )
    print(f"Accuracy: {accuracy}")

    # Calculate and print balanced accuracy score
    balanced_accuracy = balanced_accuracy_score(
        pred_data["registered"], pred_data["predicted_enrollment"]
    )
    print(f"Balanced Accuracy: {balanced_accuracy}")

    # Calculate and print F1 score
    f1 = f1_score(pred_data["registered"], pred_data["predicted_enrollment"])
    print(f"F1 Score: {f1}")

    # Calculate and print ROC AUC score
    roc_auc = roc_auc_score(pred_data["registered"], likelihood)
    print(f"ROC AUC Score: {roc_auc}")
    print("\n")


enrollment_prediction(model_OTR, pred60_data, "60", base_path)

# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()
