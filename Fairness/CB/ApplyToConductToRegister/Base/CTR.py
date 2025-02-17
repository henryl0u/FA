import pandas as pd
import numpy as np
import sys
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


base_path = "./Fairness/CB/ApplyToConductToRegister/Base"

# Save the default standard output
default_stdout = sys.stdout

# Open the file you want to redirect output to
f = open(f"{base_path}/prints.txt", "w")

# Change standard output to point to the file
sys.stdout = f

# Load data
ev_data = pd.read_excel("./Data/CTR/Training Data 58-65, Conduct To Register.xlsx")
pred66_data = pd.read_excel(f"{base_path}/data/pred66_data.xlsx")
pred67_data = pd.read_excel(f"{base_path}/data/pred67_data.xlsx")

# Fix only the country column by replacing NaN with "NA"
for df in [pred66_data, pred67_data]:
    df["country"] = df["country"].fillna("NA")

# Handle null tuition values (loss due to cost from advertising)
pred66_data["tuition"] = pred66_data["tuition"].fillna(0)
pred67_data["tuition"] = pred67_data["tuition"].fillna(0)

pred66_data.rename(columns={"tuition": "tuition_actual"}, inplace=True)
pred67_data.rename(columns={"tuition": "tuition_actual"}, inplace=True)


def replace_nulls_with_nearest_tuition(target_data, source_data):
    for index, row in target_data.iterrows():
        # Filter out rows in source_data where 'tuition' is NA
        valid_tuitions = source_data["tuition"].dropna()
        if not valid_tuitions.empty:
            # Calculate the absolute difference between 'tuition_predicted' and valid tuitions
            nearest_index = (valid_tuitions - row["tuition_predicted"]).abs().idxmin()
            # Get the nearest tuition from source_data
            nearest_tuition = source_data.at[nearest_index, "tuition"]
            # Replace the null in target_data with the nearest tuition from source_data
            target_data.at[index, "tuition"] = nearest_tuition


# Apply the function to each DataFrame, using ev_data as the source for replacements
replace_nulls_with_nearest_tuition(pred66_data, ev_data)
replace_nulls_with_nearest_tuition(pred67_data, ev_data)


# Function to randomly update 'course_completion_status' based on 'conduct_likelihood'
def update_course_completion_status(data, seed=42):
    rng = np.random.default_rng(seed)
    # Iterate over each row in the dataframe
    for index, row in data.iterrows():
        # Check if 'course_completion_status' is 'No Completion'
        if row["course_completion_status"] == "No Completion Yet":
            # Calculate the probability based on 'conduct_likelihood' * 0.4
            # 40% chance of completing the course after the interview
            probability = row["interview_likelihood"] * 0.2

            # Generate a random number between 0 and 1
            random_number = rng.random()

            # If the random number is less than or equal to the probability, update the status
            if random_number <= probability:
                data.at[index, "course_completion_status"] = "After Interview"
            else:
                data.at[index, "course_completion_status"] = "No Completion"

    return data


pred66_data = update_course_completion_status(pred66_data)
pred67_data = update_course_completion_status(pred67_data)


# Apply mapping to columns
def apply_mappings(data, mappings):
    data = data.copy()  # Create a copy to avoid warnings
    for column, mapping in mappings.items():
        if column in data.columns:  # Ensure the column exists
            data[column] = data[column].map(mapping)
    return data


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

pred66_data_copy = apply_mappings(pred66_data, mappings)
pred67_data_copy = apply_mappings(pred67_data, mappings)

# Load your pretrained models and scalers
model_CTR_58_65 = joblib.load("./Fairness/CB/ConductToRegister/Base/models/model_clf.pkl")

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
    "course_completion_status",
    "tuition",
    "management_leadership_experience",
    "tuition_benefits",
    "english_proficient",
]


def likelihood_prediction(best_clf, pred_data, output, pred_name, features, base_path):
    # Load the new prediction data
    data = pred_data

    # Check if all feature columns exist in the data
    if not all(col in data.columns for col in features):
        missing_cols = set(features) - set(data.columns)
        raise ValueError(f"Missing columns in the input data: {missing_cols}")

    X = data[features]

    # Predict the probability of enrolling (class = 1) for the new data
    likelihood = best_clf.predict_proba(X)[:, 1]

    # Collect the results in a DataFrame to keep all original features and add our predictions
    output["enrollment_likelihood"] = np.round(likelihood, 2)

    # Define the save directory path based on prediction name
    save_dir = f"{base_path}/prediction/{pred_name}"

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the likelihood predictions to an Excel file for later use
    output.to_excel(f"{save_dir}/enrollment_likelihood_{pred_name}.xlsx", index=False)


def plot_regression_diagnostics(data, save_path=None, pred_name=None):
    # Ensure the required columns exist
    required_columns = [
        "tuition",
        "tuition_actual",
        "enrollment_likelihood",
        "interview_likelihood",
        "course_completion_status",
        "CC_actual",
    ]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    def tuition_prediction(data):
        for index, row in data.iterrows():
            probability = row["enrollment_likelihood"] * row["interview_likelihood"]
            data.at[index, "EV"] = np.round(row["tuition"] * probability)
        return data

    tuition = data["tuition_actual"]
    data = tuition_prediction(data)
    pred_tuition = data["EV"]
    residuals = tuition - pred_tuition

    if pred_name:
        print(f"\nResiduals for {pred_name}:")
    print("Sum of predicted tuition (Sum of EV):", pred_tuition.sum())
    print("Sum of actual tuition:", tuition.sum())
    print("Sum of residuals:", residuals.sum())
    print(
        "Ratio of sum of residuals to sum of actual tuition:",
        np.abs(residuals.sum()) / tuition.sum() if tuition.sum() != 0 else np.nan,
    )

    results_df = pd.DataFrame(
        {"Actual": tuition, "Predicted": pred_tuition, "Residuals": residuals}
    )

    plt.figure(figsize=(12, 6))
    sns.histplot(results_df["Residuals"], bins=20, color="purple")
    plt.title("Distribution of Residuals")
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Count")
    if save_path:
        plt.savefig(f"{save_path}/residuals_histogram.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(tuition, pred_tuition, color="blue", alpha=0.6)
    plt.plot(
        [min(tuition), max(tuition)], [min(tuition), max(tuition)], color="red", lw=2
    )
    plt.title("Actual vs Predicted Tuition")
    plt.xlabel("Actual Tuition")
    plt.ylabel("Predicted Tuition")
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}/actual_vs_predicted.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(y=residuals, color="green")
    plt.title("Box Plot of Residuals")
    plt.ylabel("Residuals (Actual - Predicted)")
    if save_path:
        plt.savefig(f"{save_path}/residuals_boxplot.png")
    plt.show()

    # Columns you want to move to the front
    columns_to_move = [
        "EV",
        "tuition",
        "tuition_actual",
        "enrollment_likelihood",
        "interview_likelihood",
        "completion_likelihood",
        "course_completion_status",
        "CC_actual",
    ]

    # Create a new column order
    new_order = columns_to_move + [
        col for col in data.columns if col not in columns_to_move
    ]

    # Reorder the DataFrame
    df_reordered = data[new_order]

    # Optional: save the updated DataFrame to a new Excel file
    df_reordered.to_excel(f"{save_path}/tuition_{pred_name}.xlsx", index=False)


likelihood_prediction(
    model_CTR_58_65,
    pred66_data_copy,
    pred66_data,
    "66",
    features,
    base_path,
)
likelihood_prediction(
    model_CTR_58_65,
    pred67_data_copy,
    pred67_data,
    "67",
    features,
    base_path,
)

pred66_data = pd.read_excel(f"{base_path}/prediction/66/enrollment_likelihood_66.xlsx")
pred67_data = pd.read_excel(f"{base_path}/prediction/67/enrollment_likelihood_67.xlsx")

prediction_datasets = {
    "66": pred66_data,
    "67": pred67_data,
}


for pred_name, pred_data in prediction_datasets.items():
    # Define the save directory path based on prediction name
    save_dir = f"{base_path}/prediction/{pred_name}"

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot regression diagnostics for the predictions
    plot_regression_diagnostics(
        pred_data,
        save_path=save_dir,
        pred_name=pred_name,
    )


def analyze_dataset(data, dataset_name):
    sum_ev = np.sum(data["EV"])
    num_applicants = len(data)
    ev_per_applicant = sum_ev / num_applicants
    avg_atctr_likelihood = np.mean(
        data["enrollment_likelihood"] * data["interview_likelihood"]
    )
    price = ev_per_applicant / avg_atctr_likelihood
    avg_discount = np.mean(data["base_tuition"] - data["tuition"])

    print(f"\nAnalysis for {dataset_name}:")
    print("Sum EV:", sum_ev)
    print("Num applicants:", num_applicants)
    print("EV per applicant:", ev_per_applicant)
    print("Average ATCTR likelihood:", avg_atctr_likelihood)
    print("Price:", price)
    print("Average discount:", avg_discount)


# Load your datasets
pred66_data = pd.read_excel(f"{base_path}/prediction/66/tuition_66.xlsx")
pred67_data = pd.read_excel(f"{base_path}/prediction/67/tuition_67.xlsx")

# Analyze datasets using the function with dataset names
analyze_dataset(pred66_data, "Prediction 66")
analyze_dataset(pred67_data, "Prediction 67")

# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()
