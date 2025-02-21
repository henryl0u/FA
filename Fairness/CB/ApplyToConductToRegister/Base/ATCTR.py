import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import os
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

base_path = "./Fairness/CB/ApplyToConductToRegister/Base"

# Save the default standard output
default_stdout = sys.stdout

# Open the file you want to redirect output to
f = open(f"{base_path}/prints.txt", "w")

# Change standard output to point to the file
sys.stdout = f

# Load data
training_data = pd.read_excel("./Data/FA Training Data 58-65, Conduct To Register.xlsx")
pred66_data = pd.read_excel("./Data/FA Test Data 66, Apply To Conduct To Register.xlsx")
pred67_data = pd.read_excel("./Data/FA Test Data 67, Apply To Conduct To Register.xlsx")

# Fix only the country column by replacing NaN with "NA"
for df in [pred66_data, pred67_data]:
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
replace_nulls_with_nearest_tuition(pred66_data, training_data)
replace_nulls_with_nearest_tuition(pred67_data, training_data)

# # Handle null tuition values (loss due to cost from advertising)
# pred66_data["tuition"] = pred66_data["tuition"].fillna(0)
# pred67_data["tuition"] = pred67_data["tuition"].fillna(0)

# pred66_data.rename(columns={"tuition": "tuition_actual"}, inplace=True)
# pred67_data.rename(columns={"tuition": "tuition_actual"}, inplace=True)


# def replace_nulls_with_nearest_tuition(target_data, source_data):
#     for index, row in target_data.iterrows():
#         # Filter out rows in source_data where 'tuition' is NA
#         valid_tuitions = source_data["tuition"].dropna()
#         if not valid_tuitions.empty:
#             # Calculate the absolute difference between 'tuition_predicted' and valid tuitions
#             nearest_index = (valid_tuitions - row["tuition_predicted"]).abs().idxmin()
#             # Get the nearest tuition from source_data
#             nearest_tuition = source_data.at[nearest_index, "tuition"]
#             # Replace the null in target_data with the nearest tuition from source_data
#             target_data.at[index, "tuition"] = nearest_tuition


# # Apply the function to each DataFrame, using ev_data as the source for replacements
# replace_nulls_with_nearest_tuition(pred66_data, training_data)
# replace_nulls_with_nearest_tuition(pred67_data, training_data)

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


pred66_data = apply_mappings(pred66_data, mappings)
pred67_data = apply_mappings(pred67_data, mappings)


# Load your pretrained models and scalers
model_ATC_58_65 = joblib.load("./Fairness/CB/ApplyToConduct/Base/models/model_clf.pkl")

features = [
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


pred66_data = conduct_predictions(model_ATC_58_65, pred66_data, features)
pred67_data = conduct_predictions(model_ATC_58_65, pred67_data, features)


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


def likelihood_prediction(calibrated_clf, pred_data, features):
    X_test = pred_data[features]

    likelihood_calibrated = calibrated_clf.predict_proba(X_test)[:, 1]

    likelihood_df = pred_data.copy()
    likelihood_df["enrollment_likelihood"] = likelihood_calibrated

    return likelihood_df


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
    "tuition",
    "management_leadership_experience",
    "tuition_benefits",
    "english_proficient",
]

calibrated_clf = joblib.load(f"Fairness/CB/ConductToRegister/Base/models/model_clf.pkl")

pred66_data = likelihood_prediction(
    calibrated_clf, pred66_data, feature_columns
)
pred67_data = likelihood_prediction(
    calibrated_clf, pred67_data, feature_columns
)


def enrollment_prediction(pred_data, pred_name, base_path):
    likelihood = pred_data["enrollment_likelihood"] * pred_data["interview_likelihood"]
    pred_data["predicted_enrollment"] = pred_data["enrollment_likelihood"] > 0.6
    pred_data["predicted_enrollment"] = pred_data["predicted_enrollment"].astype(bool)

    # Columns you want to move to the front
    columns_to_move = [
        "did_interview",
        "accepted_offer",
        "predicted_enrollment",
        "tuition",
        "tuition_predicted",
        #"tuition_actual",
        "enrollment_likelihood",
        "interview_likelihood",
    ]

    # Create a new column order
    new_order = columns_to_move + [
        col for col in pred_data.columns if col not in columns_to_move
    ]

    # Reorder the DataFrame
    df_reordered = pred_data[new_order]

    save_dir = f"{base_path}/prediction/{pred_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Save the likelihood predictions to an Excel file for later use
    df_reordered.to_excel(
        f"{save_dir}/enrollment_likelihood_{pred_name}.xlsx", index=False
    )

    plot_confusion_matrix(
        pred_data["accepted_offer"],
        pred_data["predicted_enrollment"],
        calibrated_clf,
        save_path=save_dir,
    )
    plot_roc_curve(pred_data["accepted_offer"], likelihood, save_path=save_dir)

    print(f"Predictions for {pred_name}:")
    # Calculate and print classification report
    print("Classification Report:")
    print(classification_report(pred_data["accepted_offer"], pred_data["predicted_enrollment"]))

    # Calculate and print accuracy score
    accuracy = accuracy_score(pred_data["accepted_offer"], pred_data["predicted_enrollment"])
    print(f"Accuracy: {accuracy}")

    # Calculate and print balanced accuracy score
    balanced_accuracy = balanced_accuracy_score(pred_data["accepted_offer"], pred_data["predicted_enrollment"])
    print(f"Balanced Accuracy: {balanced_accuracy}")

    # Calculate and print F1 score
    f1 = f1_score(pred_data["accepted_offer"], pred_data["predicted_enrollment"])
    print(f"F1 Score: {f1}")

    # Calculate and print ROC AUC score
    roc_auc = roc_auc_score(pred_data["accepted_offer"], likelihood)
    print(f"ROC AUC Score: {roc_auc}")
    print("\n")


enrollment_prediction(pred66_data, "66", base_path)
enrollment_prediction(pred67_data, "67", base_path)

# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()
