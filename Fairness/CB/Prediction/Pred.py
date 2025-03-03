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


def demographic_parity(pred_data, group_column, pred_column):
    groups = pred_data[group_column].unique()
    probabilities = {
        group: np.mean(pred_data[pred_data[group_column] == group][pred_column])
        for group in groups
    }
    return probabilities


def equal_opportunity(pred_data, group_column, target_column, pred_column):
    groups = pred_data[group_column].unique()
    tpr = {
        group: np.mean(
            pred_data[
                (pred_data[group_column] == group) & (pred_data[target_column] == 1)
            ][pred_column]
        )
        for group in groups
    }
    return tpr


def disparate_impact_ratio(pred_data, group_column, pred_column):
    groups = pred_data[group_column].unique()
    rates = {
        group: np.mean(pred_data[pred_data[group_column] == group][pred_column])
        for group in groups
    }
    base_group = min(rates, key=rates.get)
    return {
        group: rates[group] / rates[base_group]
        for group in rates
        if rates[base_group] > 0
    }


def fairness_evaluation(
    pred_data, pred_name, group_column, target_column, pred_column, base_path=None
):
    # Ensure the save directory exists
    save_dir = os.path.join(base_path, pred_name)
    os.makedirs(save_dir, exist_ok=True)

    groups = pred_data[group_column].unique()
    fairness_results = {}

    for group in groups:
        group_data = pred_data[pred_data[group_column] == group]

        accuracy = accuracy_score(group_data[target_column], group_data[pred_column])
        balanced_acc = balanced_accuracy_score(
            group_data[target_column], group_data[pred_column]
        )
        f1 = f1_score(group_data[target_column], group_data[pred_column])
        roc_auc = roc_auc_score(
            group_data[target_column],
            group_data["enrollment_likelihood"] * group_data["interview_likelihood"],
        )

        fairness_results[group] = {
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_acc,
            "F1 Score": f1,
            "ROC AUC Score": roc_auc,
            "Sample Size": len(group_data),
        }

        # Plot and save confusion matrix
        cm = confusion_matrix(group_data[target_column], group_data[pred_column])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Not Registered", "Registered"]
        )
        disp.plot()
        plt.title(f"Confusion Matrix for {group}")

        # Sanitize group name to avoid issues
        safe_group = str(group).replace("/", "_").replace(" ", "_")
        save_path = os.path.join(save_dir, f"confusion_matrix_{safe_group}.png")

        plt.savefig(save_path)
        plt.show()

    return fairness_results


# Example usage:
fairness_metrics_gender = fairness_evaluation(
    pred60_data, "60", "gender", "registered", "predicted_enrollment", base_path
)
fairness_metrics_ethnicity = fairness_evaluation(
    pred60_data, "60", "ethnicity", "registered", "predicted_enrollment", base_path
)

demographic_parity_gender = demographic_parity(
    pred60_data, "gender", "predicted_enrollment"
)
equal_opportunity_gender = equal_opportunity(
    pred60_data, "gender", "registered", "predicted_enrollment"
)
disparate_impact_gender = disparate_impact_ratio(
    pred60_data, "gender", "predicted_enrollment"
)

demographic_parity_ethnicity = demographic_parity(
    pred60_data, "ethnicity", "predicted_enrollment"
)
equal_opportunity_ethnicity = equal_opportunity(
    pred60_data, "ethnicity", "registered", "predicted_enrollment"
)
disparate_impact_ethnicity = disparate_impact_ratio(
    pred60_data, "ethnicity", "predicted_enrollment"
)

# Print results in a formatted manner
for category, metrics_dict in {
    "Gender": fairness_metrics_gender,
    "Ethnicity": fairness_metrics_ethnicity,
}.items():
    print(f"Fairness Evaluation by {category}:\n")
    # Print header
    print(f"{'Group':<20}{'Accuracy':<10}{'Balanced Accuracy':<20}{'F1 Score':<10}{'ROC AUC Score':<15}{'Sample Size'}")
    print("-" * 85)
    
    for group, metrics in metrics_dict.items():
        # Print results in a table format
        print(f"{group:<20}{metrics['Accuracy']:<10.4f}{metrics['Balanced Accuracy']:<20.4f}{metrics['F1 Score']:<10.4f}{metrics['ROC AUC Score']:<15.4f}{metrics['Sample Size']}")
    
    print("\n")

# Prepare the fairness metric results for both gender and ethnicity
gender_metrics = {
    "Gender": ["Female", "Male", "Other"],
    "Demographic Parity": [
        demographic_parity_gender["female"],
        demographic_parity_gender["male"],
        demographic_parity_gender["other"],
    ],
    "Equal Opportunity": [
        equal_opportunity_gender["female"],
        equal_opportunity_gender["male"],
        equal_opportunity_gender["other"],
    ],
    "Disparate Impact Ratio": [
        disparate_impact_gender["female"],
        disparate_impact_gender["male"],
        disparate_impact_gender["other"],
    ],
}

ethnicity_metrics = {
    "Ethnicity": [
        "African American",
        "Asian",
        "Caucasian",
        "Latin American",
        "Arab",
        "Unknown/Other",
    ],
    "Demographic Parity": [
        demographic_parity_ethnicity["African American"],
        demographic_parity_ethnicity["Asian"],
        demographic_parity_ethnicity["Caucasian"],
        demographic_parity_ethnicity["Latin American"],
        demographic_parity_ethnicity["Arab"],
        demographic_parity_ethnicity["Unknown/Other"],
    ],
    "Equal Opportunity": [
        equal_opportunity_ethnicity["African American"],
        equal_opportunity_ethnicity["Asian"],
        equal_opportunity_ethnicity["Caucasian"],
        equal_opportunity_ethnicity["Latin American"],
        equal_opportunity_ethnicity["Arab"],
        equal_opportunity_ethnicity["Unknown/Other"],
    ],
    "Disparate Impact Ratio": [
        disparate_impact_ethnicity["African American"],
        disparate_impact_ethnicity["Asian"],
        disparate_impact_ethnicity["Caucasian"],
        disparate_impact_ethnicity["Latin American"],
        disparate_impact_ethnicity["Arab"],
        disparate_impact_ethnicity["Unknown/Other"],
    ],
}

# Convert the results to DataFrames for better visualization
gender_df = pd.DataFrame(gender_metrics)
ethnicity_df = pd.DataFrame(ethnicity_metrics)

# Print the formatted results
print("Fairness Metrics by Gender:")
print(gender_df.to_string(index=False))

print("\nFairness Metrics by Ethnicity:")
print(ethnicity_df.to_string(index=False))

# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()
