import numpy as np
import sys
from scipy.stats import ttest_rel
import json

base_path = "./Fairness/SVM/base/result"

# Save the default standard output
default_stdout = sys.stdout

# Open the file you want to redirect output to
f = open(f"{base_path}/prints.txt", "w")

# Change standard output to point to the file
sys.stdout = f

accuracy = [
    0.8539289207694816,
    0.8219758721878057,
    0.8522986631887838,
    0.843495272253016,
    0.8102380176067818,
    0.8386044995109228,
    0.8353439843495273,
    0.834039778284969,
    0.8284969025105967,
    0.8379523964786436
]

balanced_accuracy = [
    0.7925655528929044,
    0.815234509697105,
    0.7865756745087698,
    0.804061758373587,
    0.8023774786078359,
    0.8039400555776102,
    0.8033829755174542,
    0.7954498367154155,
    0.8042120758150465,
    0.8056848245306474
]

f1_score = [
    0.6303630363036303,
    0.6176470588235294,
    0.6234413965087282,
    0.6290571870170015,
    0.5980662983425414,
    0.6241457858769932,
    0.6205860255447032,
    0.6129277566539924,
    0.6149341142020498,
    0.6249056603773585
]

roc_auc_score = [
    0.8897053923031675,
    0.8899654595874277,
    0.8888194249253123,
    0.8894446005975029,
    0.8894025841560347,
    0.8892011950055492,
    0.890018342349965,
    0.8894982077814448,
    0.8887897236477226,
    0.8908804038214678
]

demographic_parity_african_american = [0.208566, 0.320298, 0.227188, 0.268156, 0.333333, 0.271881, 0.277467, 0.286778, 0.281192, 0.273743]
demographic_parity_asian = [0.258373, 0.312600, 0.258373, 0.271132, 0.344498, 0.291866, 0.288676, 0.298246, 0.296651, 0.291866]
demographic_parity_caucasian = [0.226586, 0.292044, 0.210473, 0.238671, 0.289023, 0.250755, 0.255791, 0.241692, 0.275932, 0.256798]
demographic_parity_latin_american = [0.195833, 0.275000, 0.195833, 0.220833, 0.266667, 0.220833, 0.233333, 0.212500, 0.245833, 0.229167]
demographic_parity_arab = [0.172222, 0.211111, 0.155556, 0.194444, 0.194444, 0.172222, 0.188889, 0.161111, 0.211111, 0.150000]
demographic_parity_unknown = [0.177551, 0.240816, 0.177551, 0.218367, 0.242857, 0.218367, 0.222449, 0.216327, 0.224490, 0.224490]

equal_opportunity_african_american = [0.670732, 0.829268, 0.707317, 0.743902, 0.804878, 0.768293, 0.780488, 0.780488, 0.768293, 0.756098]
equal_opportunity_asian = [0.766423, 0.832117, 0.773723, 0.802920, 0.854015, 0.824818, 0.810219, 0.810219, 0.810219, 0.817518]
equal_opportunity_caucasian = [0.727778, 0.805556, 0.677778, 0.738889, 0.783333, 0.733333, 0.750000, 0.711111, 0.777778, 0.761111]
equal_opportunity_latin_american = [0.666667, 0.791667, 0.625000, 0.729167, 0.791667, 0.729167, 0.708333, 0.708333, 0.750000, 0.708333]
equal_opportunity_arab = [0.500000, 0.607143, 0.464286, 0.535714, 0.607143, 0.571429, 0.607143, 0.535714, 0.571429, 0.535714]
equal_opportunity_unknown = [0.616438, 0.808219, 0.630137, 0.726027, 0.739726, 0.712329, 0.712329, 0.698630, 0.739726, 0.739726]

predictive_parity_african_american = [0.491071, 0.395349, 0.475410, 0.423611, 0.368715, 0.431507, 0.429530, 0.415584, 0.417219, 0.421769]
predictive_parity_asian = [0.648148, 0.581633, 0.654321, 0.647059, 0.541667, 0.617486, 0.613260, 0.593583, 0.596774, 0.612022]
predictive_parity_caucasian = [0.582222, 0.500000, 0.583732, 0.561181, 0.491289, 0.530120, 0.531496, 0.533333, 0.510949, 0.537255]
predictive_parity_latin_american = [0.680851, 0.575758, 0.638298, 0.660377, 0.593750, 0.660377, 0.607143, 0.666667, 0.610169, 0.618182]
predictive_parity_arab = [0.451613, 0.447368, 0.464286, 0.428571, 0.485714, 0.516129, 0.500000, 0.517241, 0.421053, 0.555556]
predictive_parity_unknown = [0.517241, 0.500000, 0.528736, 0.495327, 0.453782, 0.485981, 0.477064, 0.481132, 0.490909, 0.490909]


def print_perf_stats(metric_name, values):
    print(f"{metric_name} - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")

print("\n=== Performance Metrics ===")
print_perf_stats("Accuracy", accuracy)
print_perf_stats("Balanced Accuracy", balanced_accuracy)
print_perf_stats("F1 Score", f1_score)
print_perf_stats("ROC AUC Score", roc_auc_score)


# === FAIRNESS METRICS ===
# Replace with your actual data
fairness_metrics = {
    "Demographic Parity": {
        "African American": demographic_parity_african_american,
        "Asian": demographic_parity_asian,
        "Caucasian": demographic_parity_caucasian,
        "Latin American": demographic_parity_latin_american,
        "Arab": demographic_parity_arab,
        "Unknown/Other": demographic_parity_unknown
    },
    "Equal Opportunity": {
        "African American": equal_opportunity_african_american,
        "Asian": equal_opportunity_asian,
        "Caucasian": equal_opportunity_caucasian,
        "Latin American": equal_opportunity_latin_american,
        "Arab": equal_opportunity_arab,
        "Unknown/Other": equal_opportunity_unknown
    },
    "Predictive Parity": {
        "African American": predictive_parity_african_american,
        "Asian": predictive_parity_asian,
        "Caucasian": predictive_parity_caucasian,
        "Latin American": predictive_parity_latin_american,
        "Arab": predictive_parity_arab,
        "Unknown/Other": predictive_parity_unknown
    }
}

reference_group = "Caucasian"

for metric_name, groups in fairness_metrics.items():
    print(f"\n=== {metric_name} ===")
    ref_values = np.array(groups[reference_group])
    
    for group, values in groups.items():
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{group}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")
    
    print(f"\n--- Difference vs Reference Group: {reference_group} ---")
    for group, values in groups.items():
        if group == reference_group:
            continue
        values = np.array(values)
        diff = values - ref_values
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        t_stat, p_val = ttest_rel(values, ref_values)
        print(f"{group} - ΔMean: {mean_diff:.4f}, ΔStd: {std_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}")

# To store all difference values
fairness_differences = {}

for metric_name, groups in fairness_metrics.items():
    ref_values = np.array(groups[reference_group])
    metric_diff = {}

    for group, values in groups.items():
        if group == reference_group:
            continue
        values = np.array(values)
        diff = (values - ref_values).tolist()
        metric_diff[group] = diff

    fairness_differences[metric_name] = metric_diff

# Save the difference dictionary to a JSON file
with open(f"{base_path}/fairness_differences.json", "w") as json_file:
    json.dump(fairness_differences, json_file, indent=4)

# === Aggregated Disparities vs Reference Group (Reweighted) ===
print("\n=== Aggregated Fairness Disparities (vs Caucasian, Reweighted) ===")

for metric_name, groups in fairness_metrics.items():
    ref_values = np.array(groups[reference_group])
    all_disparities = []

    for group, values in groups.items():
        if group == reference_group:
            continue  # Skip reference group
        values = np.array(values)
        disparities = values - ref_values
        all_disparities.extend(disparities.tolist())  # Add all disparities across folds

    all_disparities = np.array(all_disparities)
    mean_disp = np.mean(all_disparities)
    std_disp = np.std(all_disparities)
    print(f"{metric_name} Disparity (vs {reference_group}): {mean_disp:.4f} ± {std_disp:.4f}")

# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()