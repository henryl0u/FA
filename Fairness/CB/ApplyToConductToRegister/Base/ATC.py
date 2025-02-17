import pandas as pd
import sys
import joblib
import numpy as np

base_path = "./Fairness/CB/ApplyToConductToRegister/Base"

# Save the default standard output
default_stdout = sys.stdout

# Open the file you want to redirect output to
f = open(f"{base_path}/prints.txt", "w")

# Change standard output to point to the file
sys.stdout = f

pred66_data = pd.read_excel(
    "./Data/FA/FA Test Data 66, Apply To Conduct To Register.xlsx"
)
pred67_data = pd.read_excel(
    "./Data/FA/FA Test Data 67, Apply To Conduct To Register.xlsx"
)
# Fix only the country column by replacing NaN with "NA"
for df in [pred66_data, pred67_data]:
    df["country"] = df["country"].fillna("NA")

pred66_data.rename(columns={"course_completion_status": "CC_actual"}, inplace=True)
pred67_data.rename(columns={"course_completion_status": "CC_actual"}, inplace=True)


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
    data = data.copy()  # Create a copy to avoid warnings
    for column, mapping in mappings.items():
        if column in data.columns:  # Ensure the column exists
            data[column] = data[column].map(mapping)
    return data


pred66_data_copy = apply_mappings(pred66_data, mappings)
pred67_data_copy = apply_mappings(pred67_data, mappings)


# Load your pretrained models and scalers
model_CC_58_65 = joblib.load("./Fairness/CB/CourseCompletion/Base/models/model_clf.pkl")


def course_predictions(data, output, model, seed=42):
    # Select relevant columns
    X = data[
        [
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
    ]

    # Run the model's prediction (classification) on the original data
    # Pass categorical features to the predict method
    predictions = model.predict_proba(X)[:, 1]

    # Set the predicted completion likelihood
    output["completion_likelihood"] = np.round(predictions, 2)

    # Initialize random number generator with a seed
    rng = np.random.default_rng(seed)

    # Generate random numbers and set course completion status
    random_numbers = rng.random(len(predictions))  # Generate random numbers
    output["course_completion_status"] = np.where(
        random_numbers < output["completion_likelihood"],
        "Before Interview",
        "No Completion Yet",
    )

    return output


pred66_data = course_predictions(pred66_data_copy, pred66_data, model_CC_58_65)
pred67_data = course_predictions(pred67_data_copy, pred67_data, model_CC_58_65)


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
        "No Completion Yet": 1,
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
model_ATC_58_65 = joblib.load("./Fairness/CB/ApplyToConduct/Base/models/model_clf.pkl")

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
    "management_leadership_experience",
    "tuition_benefits",
    "english_proficient",
]


def conduct_predictions(data, output, model):

    # Run the model's prediction (classification) on the original data
    predictions = model.predict_proba(data)[:, 1]

    output["interview_likelihood"] = np.round(predictions, 2)

    return output


pred66_data_copy = pred66_data_copy[feature_columns]
pred67_data_copy = pred67_data_copy[feature_columns]


pred66_data = conduct_predictions(pred66_data_copy, pred66_data, model_ATC_58_65)
pred67_data = conduct_predictions(pred67_data_copy, pred67_data, model_ATC_58_65)

pred66_data.to_excel(f"{base_path}/data/pred66_data.xlsx", index=False)
pred67_data.to_excel(f"{base_path}/data/pred67_data.xlsx", index=False)
