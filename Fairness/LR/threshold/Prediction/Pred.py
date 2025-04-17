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
sys.path.append(os.getcwd())
from abroca import *

base_path = "./Fairness/LR/threshold/Prediction"

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

# Load your pretrained models and scalers
optimizer_ITC = joblib.load("./Fairness/LR/threshold/InviteToConduct/model/optimizer.pkl")

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
    optimizer,
    pred_data,
    features,
):
    X_pred = pred_data[features]
    sensitive_values = X_pred["ethnicity"]

    predictions = optimizer.predict(
        X_pred,
        sensitive_features=sensitive_values
    )

    output = pred_data.copy()
    output["interview_prediction"] = predictions

    return output


pred60_data = conduct_predictions(
    optimizer_ITC,
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


optimizer_OTR = joblib.load("./Fairness/LR/threshold/OfferToRegister/model/optimizer.pkl")

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


def likelihood_prediction(optimizer, pred_data, features):
    X_pred = pred_data[features]
    sensitive_values = X_pred["ethnicity"]

    predictions = optimizer.predict(
        X_pred,
        sensitive_features=sensitive_values
    )

    likelihood_df = pred_data.copy()
    likelihood_df["enrollment_prediction"] = predictions

    return likelihood_df


pred60_data = likelihood_prediction(optimizer_OTR, pred60_data, features_OTR)



def enrollment_prediction(pred_data, pred_name, base_path):
    pred_data["predicted_enrollment"] = (pred_data["interview_prediction"] & pred_data["enrollment_prediction"]).astype(bool)

    save_dir = f"{base_path}/{pred_name}"
    os.makedirs(save_dir, exist_ok=True)

    # plot_confusion_matrix(
    #     pred_data["registered"],
    #     pred_data["predicted_enrollment"],
    #     save_path=save_dir,
    # )
    # plot_roc_curve(pred_data["registered"], likelihood, save_path=save_dir)

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


enrollment_prediction(pred60_data, "60", base_path)


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
        group: (np.sum((pred_data[group_column] == group) & 
                       (pred_data[target_column] == 1) & 
                       (pred_data[pred_column] == 1)) / 
                np.sum((pred_data[group_column] == group) & (pred_data[target_column] == 1))
                if np.sum((pred_data[group_column] == group) & (pred_data[target_column] == 1)) > 0 else 0)
        for group in groups
    }
    
    return tpr  # No normalization, just raw TPRs

def predictive_parity(pred_data, group_column, pred_column, target_column):
    groups = pred_data[group_column].unique()
    
    ppv = {
        group: (np.sum((pred_data[group_column] == group) & 
                    (pred_data[pred_column] == 1) & 
                    (pred_data[target_column] == 1)) / 
                np.sum((pred_data[group_column] == group) & (pred_data[pred_column] == 1)))
        if np.sum((pred_data[group_column] == group) & (pred_data[pred_column] == 1)) > 0 else np.nan
        for group in groups
    }

    return ppv

def fairness_evaluation(
    pred_data,
    pred_name,
    sensitive_column,
    target_column,
    pred_column,
    base_path=None,
):
    # Ensure the save directory exists
    save_dir = os.path.join(base_path, pred_name)
    os.makedirs(save_dir, exist_ok=True)

    groups = pred_data[sensitive_column].unique()
    fairness_results = {}

    slice_dir = os.path.join(save_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)

    for group in groups:
        group_data = pred_data[pred_data[sensitive_column] == group]

        accuracy = accuracy_score(group_data[target_column], group_data[pred_column])
        balanced_acc = balanced_accuracy_score(
            group_data[target_column], group_data[pred_column]
        )
        f1 = f1_score(group_data[target_column], group_data[pred_column])

        # Create a results dictionary for this group
        group_results = {
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_acc,
            "F1 Score": f1,
            "Sample Size": len(group_data),
        }

        fairness_results[group] = group_results

        # # Plot and save confusion matrix
        # cm = confusion_matrix(group_data[target_column], group_data[pred_column])
        # disp = ConfusionMatrixDisplay(
        #     confusion_matrix=cm, display_labels=["Not Registered", "Registered"]
        # )
        # disp.plot()
        # plt.title(f"Confusion Matrix for {group}")

        # # Sanitize group name to avoid issues
        # cm_dir = os.path.join(save_dir, "cm")
        # os.makedirs(cm_dir, exist_ok=True)
        # fig_name = str(group).replace("/", "_").replace(" ", "_")
        # save_path = os.path.join(cm_dir, f"confusion_matrix_{fig_name}.png")

        # plt.savefig(save_path)
        # plt.show()

    return fairness_results

fairness_metrics_ethnicity = fairness_evaluation(
    pred60_data,
    "60",
    "ethnicity",
    "registered",
    "predicted_enrollment",
    base_path,
)

demographic_parity_ethnicity = demographic_parity(
    pred60_data, "ethnicity", "predicted_enrollment"
)
equal_opportunity_ethnicity = equal_opportunity(
    pred60_data, "ethnicity", "registered", "predicted_enrollment"
)
predictive_parity_ethnicity = predictive_parity(
    pred60_data, "ethnicity", "predicted_enrollment", "registered"
)

# Print results in a formatted manner
for category, metrics_dict in {
    "Ethnicity": fairness_metrics_ethnicity,
}.items():
    print(f"Fairness Evaluation by {category}:")
    print(
        f"{'Group':<20}{'Accuracy':<10}{'Balanced Accuracy':<20}{'F1 Score':<10}{'Sample Size'}"
    )
    print("-" * 100)

    for group, metrics in metrics_dict.items():
        print(
            f"{group:<20}{metrics['Accuracy']:<10.4f}{metrics['Balanced Accuracy']:<20.4f}{metrics['F1 Score']:<10.4f}{metrics['Sample Size']}"
        )

# Prepare the fairness metric results for both gender and ethnicity

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
    "Predictive Parity": [
        predictive_parity_ethnicity["African American"],
        predictive_parity_ethnicity["Asian"],
        predictive_parity_ethnicity["Caucasian"],
        predictive_parity_ethnicity["Latin American"],
        predictive_parity_ethnicity["Arab"],
        predictive_parity_ethnicity["Unknown/Other"],
    ],
}

# Convert dictionary to DataFrame
ethnicity_df = pd.DataFrame(ethnicity_metrics)

# Display results
print("\nFairness Scores and Weights:")
# Ensure all columns are fully visible when printed
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.expand_frame_repr", False)  # Prevent wrapping

# Print the full DataFrame
print(ethnicity_df)



# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()
