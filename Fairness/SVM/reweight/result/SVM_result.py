import numpy as np
import sys
from scipy.stats import ttest_rel
import json

base_path = "./Fairness/SVM/reweight/result"

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

# Reweighted Model Metrics (New Batch)






accuracy_reweighted = [
    0.8294750570590154, 0.8399087055754809, 0.8395826540593414,
    0.8265405934137594, 0.8275187479621781, 0.8418650146723182,
    0.8278447994783176, 0.8301271600912944, 0.8275187479621781,
    0.8288229540267362
]

balanced_accuracy_reweighted = [
    0.8019518810326192, 0.802592269554307, 0.7881154322042985,
    0.8051628789086156, 0.7957635111836177, 0.7916466243411387,
    0.8066707620623408, 0.7959236083140395, 0.8064722705974738,
    0.7965574770430857
]

f1_score_reweighted = [
    0.6140221402214022, 0.6243305279265493, 0.61198738170347,
    0.613933236574746, 0.607275426874536, 0.6172059984214681,
    0.6162790697674418, 0.6097378277153558, 0.615831517792302,
    0.609084139985108
]

roc_auc_score_reweighted = [
    0.8890664526242891, 0.886956213072619, 0.8870808135542142,
    0.887671216998983, 0.8868330614338327, 0.8871518068518675,
    0.8866925236813356, 0.8862441068318734, 0.8869880876144223,
    0.8888078341828383
]

demographic_parity_african_american_reweighted = [0.271881, 0.249534, 0.255121, 0.296089, 0.288641, 0.245810, 0.294227, 0.284916, 0.277467, 0.275605]
demographic_parity_asian_reweighted = [0.293461, 0.279107, 0.269537, 0.288676, 0.296651, 0.279107, 0.296651, 0.285486, 0.304625, 0.290271]
demographic_parity_caucasian_reweighted = [0.267875, 0.255791, 0.231621, 0.266868, 0.258812, 0.229607, 0.262840, 0.246727, 0.274924, 0.258812]
demographic_parity_latin_american_reweighted = [0.241667, 0.241667, 0.220833, 0.258333, 0.241667, 0.225000, 0.270833, 0.233333, 0.275000, 0.250000]
demographic_parity_arab_reweighted = [0.233333, 0.227778, 0.183333, 0.250000, 0.216667, 0.200000, 0.216667, 0.216667, 0.233333, 0.194444]
demographic_parity_unknown_reweighted = [0.226531, 0.197959, 0.200000, 0.240816, 0.212245, 0.191837, 0.242857, 0.234694, 0.220408, 0.230612]

equal_opportunity_african_american_reweighted = [0.756098, 0.731707, 0.707317, 0.768293, 0.768293, 0.731707, 0.792683, 0.768293, 0.743902, 0.756098]
equal_opportunity_asian_reweighted = [0.817518, 0.788321, 0.788321, 0.802920, 0.810219, 0.795620, 0.810219, 0.788321, 0.832117, 0.802920]
equal_opportunity_caucasian_reweighted = [0.772222, 0.761111, 0.705556, 0.772222, 0.744444, 0.700000, 0.766667, 0.722222, 0.777778, 0.738889]
equal_opportunity_latin_american_reweighted = [0.708333, 0.729167, 0.645833, 0.770833, 0.729167, 0.687500, 0.770833, 0.708333, 0.812500, 0.729167]
equal_opportunity_arab_reweighted = [0.607143, 0.607143, 0.500000, 0.607143, 0.571429, 0.571429, 0.571429, 0.607143, 0.607143, 0.535714]
equal_opportunity_unknown_reweighted = [0.712329, 0.698630, 0.684932, 0.780822, 0.684932, 0.643836, 0.780822, 0.753425, 0.726027, 0.739726]

predictive_parity_african_american_reweighted = [0.424658, 0.447761, 0.423358, 0.396226, 0.406452, 0.454545, 0.411392, 0.411765, 0.409396, 0.418919]
predictive_parity_asian_reweighted = [0.608696, 0.617143, 0.639053, 0.607735, 0.596774, 0.622857, 0.596774, 0.603352, 0.596859, 0.604396]
predictive_parity_caucasian_reweighted = [0.522556, 0.539370, 0.552174, 0.524528, 0.521401, 0.552632, 0.528736, 0.530612, 0.512821, 0.517510]
predictive_parity_latin_american_reweighted = [0.586207, 0.603448, 0.584906, 0.596774, 0.603448, 0.611111, 0.569231, 0.607143, 0.590909, 0.583333]
predictive_parity_arab_reweighted = [0.404762, 0.414634, 0.424242, 0.377778, 0.410256, 0.444444, 0.410256, 0.435897, 0.404762, 0.428571]
predictive_parity_unknown_reweighted = [0.468468, 0.525773, 0.510204, 0.483051, 0.480769, 0.500000, 0.478992, 0.478261, 0.490741, 0.477876]




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

reference_group = "Caucasian"

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
with open("./Fairness/SVM/base/result/fairness_differences.json") as f:
    baseline_diffs = json.load(f)

with open("./Fairness/SVM/reweight/result/fairness_differences.json") as f:
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



# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()