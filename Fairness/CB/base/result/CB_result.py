import numpy as np
import sys
from scipy.stats import ttest_rel
import json

base_path = "./Fairness/CB/base/result"

# Save the default standard output
default_stdout = sys.stdout

# Open the file you want to redirect output to
f = open(f"{base_path}/prints.txt", "w")

# Change standard output to point to the file
sys.stdout = f

accuracy = [
    0.8487120965112488,
    0.8317574176719922,
    0.8317574176719922,
    0.844147375285295,
    0.8359960873818063,
    0.844147375285295,
    0.820345614607108,
    0.831105314639713,
    0.8167590479295729,
    0.8346918813172481
]

balanced_accuracy = [
    0.8108072082827446,
    0.809766576935002,
    0.8104804942292592,
    0.812311831540149,
    0.8059217103299594,
    0.805172658597578,
    0.8092446313129703,
    0.8172226842420959,
    0.8163421500247752,
    0.8129808347073193
]

f1_score = [
    0.639751552795031,
    0.6222547584187409,
    0.6228070175438597,
    0.6362252663622526,
    0.6232209737827715,
    0.6306027820710973,
    0.6116983791402396,
    0.6273381294964029,
    0.614010989010989,
    0.6274797942689199
]

roc_auc_score = [
    0.8982876126837495,
    0.8988446927439054,
    0.8985136321619922,
    0.897450181540004,
    0.897992048750663,
    0.8992300849311653,
    0.8974146848911774,
    0.9009187112253444,
    0.8983687478810674,
    0.9007506454594715
]

demographic_parity_african_american = [0.227188, 0.271881, 0.266294, 0.253259, 0.245810, 0.232775, 0.283054, 0.284916, 0.290503, 0.266294]
demographic_parity_asian = [0.267943, 0.293461, 0.290271, 0.275917, 0.293461, 0.282297, 0.314195, 0.295056, 0.314195, 0.291866]
demographic_parity_caucasian = [0.250755, 0.272910, 0.283988, 0.255791, 0.268882, 0.252769, 0.288016, 0.280967, 0.312185, 0.280967]
demographic_parity_latin_american = [0.233333, 0.258333, 0.245833, 0.225000, 0.237500, 0.229167, 0.270833, 0.275000, 0.275000, 0.241667]
demographic_parity_arab = [0.200000, 0.233333, 0.222222, 0.222222, 0.216667, 0.177778, 0.250000, 0.227778, 0.272222, 0.227778]
demographic_parity_unknown = [0.222449, 0.230612, 0.232653, 0.222449, 0.220408, 0.216327, 0.257143, 0.240816, 0.265306, 0.222449]

equal_opportunity_african_american = [0.731707, 0.756098, 0.780488, 0.768293, 0.743902, 0.719512, 0.756098, 0.792683, 0.804878, 0.743902]
equal_opportunity_asian = [0.781022, 0.817518, 0.781022, 0.810219, 0.795620, 0.802920, 0.824818, 0.810219, 0.824818, 0.810219]
equal_opportunity_caucasian = [0.761111, 0.788889, 0.800000, 0.761111, 0.777778, 0.766667, 0.800000, 0.805556, 0.833333, 0.805556]
equal_opportunity_latin_american = [0.729167, 0.812500, 0.770833, 0.687500, 0.750000, 0.708333, 0.812500, 0.854167, 0.812500, 0.770833]
equal_opportunity_arab = [0.535714, 0.535714, 0.571429, 0.678571, 0.607143, 0.464286, 0.642857, 0.607143, 0.678571, 0.642857]
equal_opportunity_unknown = [0.794521, 0.753425, 0.794521, 0.753425, 0.726027, 0.739726, 0.794521, 0.780822, 0.821918, 0.753425]

predictive_parity_african_american = [0.491803, 0.424658, 0.447552, 0.463235, 0.462121, 0.472000, 0.407895, 0.424837, 0.423077, 0.426573]
predictive_parity_asian = [0.636905, 0.608696, 0.587912, 0.641618, 0.592391, 0.621469, 0.573604, 0.600000, 0.573604, 0.606557]
predictive_parity_caucasian = [0.550201, 0.523985, 0.510638, 0.539370, 0.524345, 0.549801, 0.503497, 0.519713, 0.483871, 0.519713]
predictive_parity_latin_american = [0.625000, 0.629032, 0.627119, 0.611111, 0.631579, 0.618182, 0.600000, 0.621212, 0.590909, 0.637931]
predictive_parity_arab = [0.416667, 0.357143, 0.400000, 0.475000, 0.435897, 0.406250, 0.400000, 0.414634, 0.387755, 0.439024]
predictive_parity_unknown = [0.532110, 0.486726, 0.508772, 0.504587, 0.490741, 0.509434, 0.460317, 0.483051, 0.461538, 0.504587]


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