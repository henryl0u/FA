import numpy as np
import sys
from scipy.stats import ttest_rel
import json

base_path = "./Fairness/CB/reweight/result"

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

# Reweighted Model Metrics

accuracy_reweighted = [
    0.7887186175415716,
    0.8187153570264102,
    0.8037169872839909,
    0.856537332898598,
    0.8281708509944571,
    0.8350179328333877,
    0.8284969025105967,
    0.831105314639713,
    0.8366481904140854,
    0.8327355722204108
]

balanced_accuracy_reweighted = [
    0.8114084780485826,
    0.8175330988139773,
    0.8155416643726655,
    0.8091457477912392,
    0.8140084264697786,
    0.8146071607607004,
    0.8199182562887022,
    0.8136530977708104,
    0.8177413699678067,
    0.8160733896836596
]

f1_score_reweighted = [
    0.5888324873096447,
    0.616551724137931,
    0.6029023746701847,
    0.6468699839486356,
    0.6222222222222222,
    0.6290322580645161,
    0.6269503546099291,
    0.6246376811594203,
    0.6329670329670329,
    0.6279912980420594
]

roc_auc_score_reweighted = [
    0.9013309070045754,
    0.9005528784160092,
    0.9006876207972693,
    0.9003427962086683,
    0.9021625427770839,
    0.9011635656601072,
    0.9006557462554657,
    0.9014026247236332,
    0.8987838413459172,
    0.900087075452836
]

# Demographic Parity (Reweighted)
demographic_parity_african_american_reweighted = [0.351955, 0.281192, 0.316574, 0.227188, 0.264432, 0.256983, 0.271881, 0.262570, 0.260708, 0.256983]
demographic_parity_asian_reweighted = [0.352472, 0.317384, 0.328549, 0.253589, 0.307815, 0.291866, 0.306220, 0.296651, 0.296651, 0.295056]
demographic_parity_caucasian_reweighted = [0.344411, 0.314199, 0.333333, 0.233635, 0.296073, 0.275932, 0.289023, 0.280967, 0.276939, 0.290030]
demographic_parity_latin_american_reweighted = [0.308333, 0.279167, 0.300000, 0.212500, 0.254167, 0.258333, 0.279167, 0.266667, 0.250000, 0.266667]
demographic_parity_arab_reweighted = [0.305556, 0.283333, 0.283333, 0.183333, 0.238889, 0.233333, 0.255556, 0.238889, 0.222222, 0.238889]
demographic_parity_unknown_reweighted = [0.300000, 0.248980, 0.281633, 0.206122, 0.232653, 0.238776, 0.253061, 0.242857, 0.236735, 0.230612]

# Equal Opportunity (Reweighted)
equal_opportunity_african_american_reweighted = [0.853659, 0.768293, 0.829268, 0.743902, 0.768293, 0.768293, 0.768293, 0.768293, 0.768293, 0.768293]
equal_opportunity_asian_reweighted = [0.861314, 0.832117, 0.839416, 0.773723, 0.810219, 0.802920, 0.817518, 0.802920, 0.802920, 0.810219]
equal_opportunity_caucasian_reweighted = [0.838889, 0.844444, 0.850000, 0.750000, 0.816667, 0.794444, 0.822222, 0.794444, 0.811111, 0.811111]
equal_opportunity_latin_american_reweighted = [0.854167, 0.854167, 0.854167, 0.666667, 0.812500, 0.812500, 0.854167, 0.833333, 0.812500, 0.812500]
equal_opportunity_arab_reweighted = [0.750000, 0.714286, 0.678571, 0.535714, 0.642857, 0.607143, 0.678571, 0.642857, 0.607143, 0.607143]
equal_opportunity_unknown_reweighted = [0.863014, 0.780822, 0.835616, 0.739726, 0.767123, 0.780822, 0.808219, 0.780822, 0.780822, 0.780822]

# Predictive Parity (Reweighted)
predictive_parity_african_american_reweighted = [0.370370, 0.417219, 0.400000, 0.500000, 0.443662, 0.456522, 0.431507, 0.446809, 0.450000, 0.456522]
predictive_parity_asian_reweighted = [0.533937, 0.572864, 0.558252, 0.666667, 0.575130, 0.601093, 0.583333, 0.591398, 0.591398, 0.600000]
predictive_parity_caucasian_reweighted = [0.441520, 0.487179, 0.462236, 0.581897, 0.500000, 0.521898, 0.515679, 0.512545, 0.530909, 0.506944]
predictive_parity_latin_american_reweighted = [0.554054, 0.611940, 0.569444, 0.627451, 0.639344, 0.629032, 0.611940, 0.625000, 0.650000, 0.609375]
predictive_parity_arab_reweighted = [0.381818, 0.392157, 0.372549, 0.454545, 0.418605, 0.404762, 0.413043, 0.418605, 0.425000, 0.395349]
predictive_parity_unknown_reweighted = [0.428571, 0.467213, 0.442029, 0.534653, 0.491228, 0.487179, 0.475806, 0.478992, 0.491379, 0.504425]


def print_perf_stats(metric_name, values):
    print(f"{metric_name} - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")

# Print performance stats for reweighted metrics
print("\n=== Reweighted Performance Metrics ===")
print_perf_stats("Accuracy (Reweighted)", accuracy_reweighted)
print_perf_stats("Balanced Accuracy (Reweighted)", balanced_accuracy_reweighted)
print_perf_stats("F1 Score (Reweighted)", f1_score_reweighted)
print_perf_stats("ROC AUC Score (Reweighted)", roc_auc_score_reweighted)

def compare_and_print(metric_name, baseline_values, reweighted_values):
    baseline_values = np.array(baseline_values)
    reweighted_values = np.array(reweighted_values)
    diff = reweighted_values - baseline_values
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    t_stat, p_val = ttest_rel(reweighted_values, baseline_values)
    print(f"{metric_name} - ΔMean: {mean_diff:.4f}, ΔStd: {std_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}")

print("\n=== Performance Metrics Comparison ===")
compare_and_print("Accuracy", accuracy, accuracy_reweighted)
compare_and_print("Balanced Accuracy", balanced_accuracy, balanced_accuracy_reweighted)
compare_and_print("F1 Score", f1_score, f1_score_reweighted)
compare_and_print("ROC AUC Score", roc_auc_score, roc_auc_score_reweighted)


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

# Now define reweighted fairness metrics in the same structure
fairness_metrics_reweighted = {
    "Demographic Parity": {
        "African American": demographic_parity_african_american_reweighted,
        "Asian": demographic_parity_asian_reweighted,
        "Caucasian": demographic_parity_caucasian_reweighted,
        "Latin American": demographic_parity_latin_american_reweighted,
        "Arab": demographic_parity_arab_reweighted,
        "Unknown/Other": demographic_parity_unknown_reweighted
    },
    "Equal Opportunity": {
        "African American": equal_opportunity_african_american_reweighted,
        "Asian": equal_opportunity_asian_reweighted,
        "Caucasian": equal_opportunity_caucasian_reweighted,
        "Latin American": equal_opportunity_latin_american_reweighted,
        "Arab": equal_opportunity_arab_reweighted,
        "Unknown/Other": equal_opportunity_unknown_reweighted
    },
    "Predictive Parity": {
        "African American": predictive_parity_african_american_reweighted,
        "Asian": predictive_parity_asian_reweighted,
        "Caucasian": predictive_parity_caucasian_reweighted,
        "Latin American": predictive_parity_latin_american_reweighted,
        "Arab": predictive_parity_arab_reweighted,
        "Unknown/Other": predictive_parity_unknown_reweighted
    }
}

reference_group = "Caucasian"

for metric_name, groups in fairness_metrics_reweighted.items():
    print(f"\n=== {metric_name} (Reweighted) ===")
    ref_values = np.array(groups[reference_group])

    for group, values in groups.items():
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{group}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")

    print(f"\n--- Difference vs Reference Group: {metric_name} ---")
    for group, values in groups.items():
        if group == reference_group:
            continue
        values = np.array(values)
        diff = values - ref_values
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        t_stat, p_val = ttest_rel(values, ref_values)
        print(f"{group} - ΔMean: {mean_diff:.4f}, ΔStd: {std_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}")

    print(f"\n--- Difference vs Baseline Values: {metric_name} ---")
    baseline_groups = fairness_metrics[metric_name]
    for group, reweighted_values in groups.items():
        baseline_values = np.array(baseline_groups[group])
        reweighted_values = np.array(reweighted_values)
        diff = reweighted_values - baseline_values
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        t_stat, p_val = ttest_rel(reweighted_values, baseline_values)
        print(f"{group} - ΔMean: {mean_diff:.4f}, ΔStd: {std_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}")

# To store all difference values
fairness_differences = {}

for metric_name, groups in fairness_metrics_reweighted.items():
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

# Load saved disparity diffs
with open("./Fairness/CB/base/result/fairness_differences.json") as f:
    baseline_diffs = json.load(f)

with open("./Fairness/CB/reweight/result/fairness_differences.json") as f:
    mitigated_diffs = json.load(f)

# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== Disparity Comparison: {metric_name} ===")
    
    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs[metric_name][group])
        
        t_stat, p_val = ttest_rel(mitigated_diff, base_diff)

        print(f"{group} - t={t_stat:.3f}, p={p_val:.4f}")

# Helper to interpret fairness movement
def interpret_disparity_change(base_mean, delta_mean):
    if base_mean < 0 and delta_mean > 0:
        return "✅ Improved"
    elif base_mean > 0 and delta_mean < 0:
        return "✅ Improved"
    elif base_mean == 0 and delta_mean != 0:
        return "⚠️ New Disparity"
    else:
        return "❌ Worsened"

# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== ΔDiff Comparison: {metric_name} ===")
    
    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs[metric_name][group])
        
        # ΔDisparity = disparity_mitigated − disparity_base
        delta_diff = mitigated_diff - base_diff
        
        mean_base = np.mean(base_diff)
        mean_delta = np.mean(delta_diff)
        std_delta = np.std(delta_diff)
        t_stat, p_val = ttest_rel(mitigated_diff, base_diff)
        interpretation = interpret_disparity_change(mean_base, mean_delta)

        print(f"{group} - ΔΔMean: {mean_delta:.4f}, ΔΔStd: {std_delta:.4f}, t={t_stat:.3f}, p={p_val:.4f} → {interpretation}")

# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()