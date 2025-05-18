import numpy as np
import sys
from scipy.stats import ttest_rel
import json

base_path = "./Fairness/CB/egr/result"

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



# EGR Model Metrics (New Batch)

# Accuracy
accuracy_EO = [
    0.8138245842843169,
    0.8265405934137594,
    0.8255624388653408,
    0.8356700358656668,
    0.8252363873492011
]

accuracy_DP = [
    0.8066514509292468,
    0.8030648842517117,
    0.8395826540593414,
    0.8447994783175742,
    0.8461036843821323
]

# Balanced Accuracy
balanced_accuracy_EO = [
    0.8174113960180005,
    0.8172994729109859,
    0.8131344120450996,
    0.8128623918076632,
    0.8150776724630038
]

balanced_accuracy_DP = [
    0.8166141702622116,
    0.8165725160314456,
    0.8095329510320107,
    0.7970026339962273,
    0.7985105171499522
]

# F1 Score
f1_score_EO = [
    0.6123557365919892,
    0.623229461756374,
    0.6192170818505338,
    0.6283185840707964,
    0.6203966005665722
]

f1_score_DP = [
    0.6059800664451828,
    0.6031537450722734,
    0.6295180722891566,
    0.6246056782334385,
    0.627172195892575
]

# ROC AUC Score
roc_auc_EO = [
    0.8864230389188156,
    0.8888455040958786,
    0.8868700069254686,
    0.8866722398820063,
    0.8841201032735153
]

roc_auc_DP = [
    0.8809297514075509,
    0.8803849865112734,
    0.8816903938824061,
    0.8799604755681636,
    0.8789093401100542
]


# EO
demographic_parity_african_american_EO = [0.243948, 0.290503, 0.290503, 0.197393, 0.255121]
demographic_parity_asian_EO = [0.291866, 0.309410, 0.318979, 0.259968, 0.304625]
demographic_parity_caucasian_EO = [0.279960, 0.310171, 0.306143, 0.239678, 0.308157]
demographic_parity_latin_american_EO = [0.245833, 0.279167, 0.262500, 0.220833, 0.254167]
demographic_parity_arab_EO = [0.244444, 0.294444, 0.255556, 0.216667, 0.272222]
demographic_parity_unknown_EO = [0.242857, 0.269388, 0.265306, 0.224490, 0.240816]

# DP
demographic_parity_african_american_DP = [0.258845, 0.255121, 0.270019, 0.281192, 0.325885]
demographic_parity_asian_DP = [0.277512, 0.269537, 0.261563, 0.274322, 0.304625]
demographic_parity_caucasian_DP = [0.254783, 0.245720, 0.255791, 0.272910, 0.307150]
demographic_parity_latin_american_DP = [0.241667, 0.241667, 0.225000, 0.237500, 0.279167]
demographic_parity_arab_DP = [0.250000, 0.288889, 0.233333, 0.288889, 0.355556]
demographic_parity_unknown_DP = [0.248980, 0.265306, 0.271429, 0.300000, 0.353061]


# EO
equal_opportunity_african_american_EO = [0.731707, 0.792683, 0.792683, 0.597561, 0.743902]
equal_opportunity_asian_EO = [0.795620, 0.817518, 0.832117, 0.766423, 0.810219]
equal_opportunity_caucasian_EO = [0.805556, 0.838889, 0.822222, 0.722222, 0.833333]
equal_opportunity_latin_american_EO = [0.812500, 0.875000, 0.875000, 0.708333, 0.812500]
equal_opportunity_arab_EO = [0.714286, 0.750000, 0.642857, 0.535714, 0.750000]
equal_opportunity_unknown_EO = [0.780822, 0.794521, 0.821918, 0.739726, 0.753425]

# DP
equal_opportunity_african_american_DP = [0.768293, 0.743902, 0.768293, 0.780488, 0.829268]
equal_opportunity_asian_DP = [0.788321, 0.788321, 0.751825, 0.802920, 0.810219]
equal_opportunity_caucasian_DP = [0.755556, 0.750000, 0.772222, 0.794444, 0.822222]
equal_opportunity_latin_american_DP = [0.791667, 0.729167, 0.729167, 0.770833, 0.854167]
equal_opportunity_arab_DP = [0.678571, 0.750000, 0.607143, 0.714286, 0.785714]
equal_opportunity_unknown_DP = [0.780822, 0.808219, 0.849315, 0.849315, 0.904110]


# EO
predictive_parity_african_american_EO = [0.458015, 0.416667, 0.416667, 0.462264, 0.445255]
predictive_parity_asian_EO = [0.595628, 0.577320, 0.570000, 0.644172, 0.581152]
predictive_parity_caucasian_EO = [0.521583, 0.490260, 0.486842, 0.546218, 0.490196]
predictive_parity_latin_american_EO = [0.661017, 0.626866, 0.666667, 0.641509, 0.639344]
predictive_parity_arab_EO = [0.454545, 0.396226, 0.391304, 0.384615, 0.428571]
predictive_parity_unknown_EO = [0.478992, 0.439394, 0.461538, 0.490909, 0.466102]

# DP
predictive_parity_african_american_DP = [0.453237, 0.445255, 0.434483, 0.423841, 0.388571]
predictive_parity_asian_DP = [0.620690, 0.639053, 0.628049, 0.639535, 0.581152]
predictive_parity_caucasian_DP = [0.537549, 0.553279, 0.547244, 0.527675, 0.485246]
predictive_parity_latin_american_DP = [0.655172, 0.603448, 0.648148, 0.649123, 0.611940]
predictive_parity_arab_DP = [0.422222, 0.403846, 0.404762, 0.384615, 0.343750]
predictive_parity_unknown_DP = [0.467213, 0.453846, 0.466165, 0.421769, 0.381503]




def print_perf_stats(metric_name, values):
    print(f"{metric_name} - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")


# Print performance stats for egr metrics
print("\n=== EGR Performance Metrics ===")
print_perf_stats("Accuracy (EO)", accuracy_EO)
print_perf_stats("Balanced Accuracy (EO)", balanced_accuracy_EO)
print_perf_stats("F1 Score (EO)", f1_score_EO)
print_perf_stats("ROC AUC (EO)", roc_auc_EO)
print_perf_stats("Accuracy (DP)", accuracy_DP)
print_perf_stats("Balanced Accuracy (DP)", balanced_accuracy_DP)
print_perf_stats("F1 Score (DP)", f1_score_DP)
print_perf_stats("ROC AUC (DP)", roc_auc_DP)


def compare_and_print(metric_name, baseline_values, egr_values):
    baseline_values = np.array(baseline_values)
    egr_values = np.array(egr_values)
    egr_values = np.concatenate((egr_values, egr_values))
    diff = egr_values - baseline_values
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    t_stat, p_val = ttest_rel(egr_values, baseline_values)
    print(
        f"{metric_name} - ΔMean: {mean_diff:.4f}, ΔStd: {std_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}"
    )


print("\n=== Performance Metrics Comparison ===")
compare_and_print("Accuracy (EO)", accuracy, accuracy_EO)
compare_and_print("Balanced Accuracy (EO)", balanced_accuracy, balanced_accuracy_EO)
compare_and_print("F1 Score (EO)", f1_score, f1_score_EO)
compare_and_print("ROC AUC (EO)", roc_auc_score, roc_auc_EO)
compare_and_print("Accuracy (DP)", accuracy, accuracy_DP)
compare_and_print("Balanced Accuracy (DP)", balanced_accuracy, balanced_accuracy_DP)
compare_and_print("F1 Score (DP)", f1_score, f1_score_DP)
compare_and_print("ROC AUC (DP)", roc_auc_score, roc_auc_DP)

# === FAIRNESS METRICS ===
# Replace with your actual data
fairness_metrics = {
    "Demographic Parity": {
        "African American": demographic_parity_african_american,
        "Asian": demographic_parity_asian,
        "Caucasian": demographic_parity_caucasian,
        "Latin American": demographic_parity_latin_american,
        "Arab": demographic_parity_arab,
        "Unknown/Other": demographic_parity_unknown,
    },
    "Equal Opportunity": {
        "African American": equal_opportunity_african_american,
        "Asian": equal_opportunity_asian,
        "Caucasian": equal_opportunity_caucasian,
        "Latin American": equal_opportunity_latin_american,
        "Arab": equal_opportunity_arab,
        "Unknown/Other": equal_opportunity_unknown,
    },
    "Predictive Parity": {
        "African American": predictive_parity_african_american,
        "Asian": predictive_parity_asian,
        "Caucasian": predictive_parity_caucasian,
        "Latin American": predictive_parity_latin_american,
        "Arab": predictive_parity_arab,
        "Unknown/Other": predictive_parity_unknown,
    },
}

# Now define egr fairness metrics in the same structure
fairness_metrics_EO = {
    "Demographic Parity": {
        "African American": demographic_parity_african_american_EO,
        "Asian": demographic_parity_asian_EO,
        "Caucasian": demographic_parity_caucasian_EO,
        "Latin American": demographic_parity_latin_american_EO,
        "Arab": demographic_parity_arab_EO,
        "Unknown/Other": demographic_parity_unknown_EO,
    },
    "Equal Opportunity": {
        "African American": equal_opportunity_african_american_EO,
        "Asian": equal_opportunity_asian_EO,
        "Caucasian": equal_opportunity_caucasian_EO,
        "Latin American": equal_opportunity_latin_american_EO,
        "Arab": equal_opportunity_arab_EO,
        "Unknown/Other": equal_opportunity_unknown_EO,
    },
    "Predictive Parity": {
        "African American": predictive_parity_african_american_EO,
        "Asian": predictive_parity_asian_EO,
        "Caucasian": predictive_parity_caucasian_EO,
        "Latin American": predictive_parity_latin_american_EO,
        "Arab": predictive_parity_arab_EO,
        "Unknown/Other": predictive_parity_unknown_EO,
    },
}

fairness_metrics_DP = {
    "Demographic Parity": {
        "African American": demographic_parity_african_american_DP,
        "Asian": demographic_parity_asian_DP,
        "Caucasian": demographic_parity_caucasian_DP,
        "Latin American": demographic_parity_latin_american_DP,
        "Arab": demographic_parity_arab_DP,
        "Unknown/Other": demographic_parity_unknown_DP,
    },
    "Equal Opportunity": {
        "African American": equal_opportunity_african_american_DP,
        "Asian": equal_opportunity_asian_DP,
        "Caucasian": equal_opportunity_caucasian_DP,
        "Latin American": equal_opportunity_latin_american_DP,
        "Arab": equal_opportunity_arab_DP,
        "Unknown/Other": equal_opportunity_unknown_DP,
    },
    "Predictive Parity": {
        "African American": predictive_parity_african_american_DP,
        "Asian": predictive_parity_asian_DP,
        "Caucasian": predictive_parity_caucasian_DP,
        "Latin American": predictive_parity_latin_american_DP,
        "Arab": predictive_parity_arab_DP,
        "Unknown/Other": predictive_parity_unknown_DP,
    },
}
reference_group = "Caucasian"

for metric_name, groups in fairness_metrics_DP.items():
    print(f"\n=== {metric_name} (EGR-DP) ===")
    ref_values = np.array(groups[reference_group])

    for group, values in groups.items():
        values = np.array(values)
        values = np.concatenate((values, values))
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
    for group, egr_values in groups.items():
        baseline_values = np.array(baseline_groups[group])
        egr_values = np.array(egr_values)
        egr_values = np.concatenate((egr_values, egr_values))
        diff = egr_values - baseline_values
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        t_stat, p_val = ttest_rel(egr_values, baseline_values)
        print(
            f"{group} - ΔMean: {mean_diff:.4f}, ΔStd: {std_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}"
        )

for metric_name, groups in fairness_metrics_EO.items():
    print(f"\n=== {metric_name} (EGR-EO) ===")
    ref_values = np.array(groups[reference_group])

    for group, values in groups.items():
        values = np.array(values)
        values = np.concatenate((values, values))
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
    for group, egr_values in groups.items():
        baseline_values = np.array(baseline_groups[group])
        egr_values = np.array(egr_values)
        egr_values = np.concatenate((egr_values, egr_values))
        diff = egr_values - baseline_values
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        t_stat, p_val = ttest_rel(egr_values, baseline_values)
        print(
            f"{group} - ΔMean: {mean_diff:.4f}, ΔStd: {std_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}"
        )

# To store all difference values
fairness_differences_DP = {}


for metric_name, groups in fairness_metrics_DP.items():
    ref_values = np.array(groups[reference_group])
    metric_diff = {}

    for group, values in groups.items():
        if group == reference_group:
            continue
        values = np.array(values)
        diff = (values - ref_values).tolist()
        metric_diff[group] = diff

    fairness_differences_DP[metric_name] = metric_diff

# Save the difference dictionary to a JSON file
with open(f"{base_path}/fairness_differences_DP.json", "w") as json_file:
    json.dump(fairness_differences_DP, json_file, indent=4)

# To store all difference values
fairness_differences_EO = {}

for metric_name, groups in fairness_metrics_EO.items():
    ref_values = np.array(groups[reference_group])
    metric_diff = {}

    for group, values in groups.items():
        if group == reference_group:
            continue
        values = np.array(values)
        diff = (values - ref_values).tolist()
        metric_diff[group] = diff

    fairness_differences_EO[metric_name] = metric_diff

# Save the difference dictionary to a JSON file
with open(f"{base_path}/fairness_differences_EO.json", "w") as json_file:
    json.dump(fairness_differences_EO, json_file, indent=4)

# Load saved disparity diffs
with open("./Fairness/CB/base/result/fairness_differences.json") as f:
    baseline_diffs = json.load(f)

with open("./Fairness/CB/egr/result/fairness_differences_DP.json") as f:
    mitigated_diffs_DP = json.load(f)

with open("./Fairness/CB/egr/result/fairness_differences_EO.json") as f:
    mitigated_diffs_EO = json.load(f)

# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== Disparity Comparison DP: {metric_name} ===")
    
    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs_DP[metric_name][group])
        mitigated_diff = np.concatenate((mitigated_diff, mitigated_diff))
        
        t_stat, p_val = ttest_rel(mitigated_diff, base_diff)

        print(f"{group} - t={t_stat:.3f}, p={p_val:.4f}")

# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== Disparity Comparison EO: {metric_name} ===")
    
    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs_EO[metric_name][group])
        mitigated_diff = np.concatenate((mitigated_diff, mitigated_diff))
        
        t_stat, p_val = ttest_rel(mitigated_diff, base_diff)

        print(f"{group} - t={t_stat:.3f}, p={p_val:.4f}")

# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== Disparity Comparison DP (ABS): {metric_name} ===")
    
    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs_DP[metric_name][group])
        mitigated_diff = np.concatenate((mitigated_diff, mitigated_diff))
        
        t_stat, p_val = ttest_rel(np.abs(mitigated_diff), np.abs(base_diff))

        print(f"{group} - t={t_stat:.3f}, p={p_val:.4f}")

# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== Disparity Comparison EO (ABS): {metric_name} ===")
    
    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs_EO[metric_name][group])
        mitigated_diff = np.concatenate((mitigated_diff, mitigated_diff))
        
        t_stat, p_val = ttest_rel(np.abs(mitigated_diff), np.abs(base_diff))

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

print("\n=== Constraint: DP ===")
# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== ΔDiff Comparison: {metric_name} ===")
    
    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs_DP[metric_name][group])
        mitigated_diff = np.concatenate((mitigated_diff, mitigated_diff))
        
        # ΔDisparity = disparity_mitigated − disparity_base
        delta_diff = mitigated_diff - base_diff
        
        mean_base = np.mean(base_diff)
        mean_delta = np.mean(delta_diff)
        std_delta = np.std(delta_diff)
        t_stat, p_val = ttest_rel(mitigated_diff, base_diff)
        interpretation = interpret_disparity_change(mean_base, mean_delta)

        print(f"{group} - ΔΔMean: {mean_delta:.4f}, ΔΔStd: {std_delta:.4f}, t={t_stat:.3f}, p={p_val:.4f} → {interpretation}")

print("\n=== Constraint: EO ===")
print("\n=== ΔDiff Comparison: Equal Opportunity ===")
# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== ΔDiff Comparison: {metric_name} ===")
    
    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs_EO[metric_name][group])
        mitigated_diff = np.concatenate((mitigated_diff, mitigated_diff))
        
        # ΔDisparity = disparity_mitigated − disparity_base
        delta_diff = mitigated_diff - base_diff
        
        mean_base = np.mean(base_diff)
        mean_delta = np.mean(delta_diff)
        std_delta = np.std(delta_diff)
        t_stat, p_val = ttest_rel(mitigated_diff, base_diff)
        interpretation = interpret_disparity_change(mean_base, mean_delta)

        print(f"{group} - ΔΔMean: {mean_delta:.4f}, ΔΔStd: {std_delta:.4f}, t={t_stat:.3f}, p={p_val:.4f} → {interpretation}")

# === Aggregated Disparities vs Reference Group (Reweighted) ===
print("\n=== Aggregated Fairness Disparities (vs Caucasian, Reweighted) ===")

for metric_name, groups in fairness_metrics_EO.items():
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
    print(
        f"{metric_name} EQ Disparity (vs {reference_group}): {mean_disp:.4f} ± {std_disp:.4f}"
    )

for metric_name, groups in fairness_metrics_DP.items():
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
    print(
        f"{metric_name} DP Disparity (vs {reference_group}): {mean_disp:.4f} ± {std_disp:.4f}"
    )

# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()