import numpy as np
import sys
from scipy.stats import ttest_rel

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
    0.8236061297685034,
    0.8451255298337137,
    0.8402347570916204,
    0.8226279752200848,
    0.8128464297358983,
    0.8343658298011086,
    0.8121943267036191,
    0.8327355722204108,
    0.8216498206716661,
    0.8278447994783176
]

balanced_accuracy_reweighted = [
    0.7905259444281851,
    0.7900619525185235,
    0.7792314903086904,
    0.7906443873278413,
    0.8011097411497436,
    0.782083899589398,
    0.8028545101027809,
    0.7875166979133765,
    0.7986159204643252,
    0.7888228297059139
]

f1_score_reweighted = [
    0.5995558845299778,
    0.6190858059342422,
    0.6048387096774194,
    0.5988200589970502,
    0.5991620111731844,
    0.6018808777429467,
    0.6,
    0.605080831408776,
    0.6044830079537238,
    0.6018099547511312
]

roc_auc_score_reweighted = [
    0.8889925616410173,
    0.8890512397747917,
    0.8873886926511795,
    0.8875422699889598,
    0.8871844058150754,
    0.8867193272733068,
    0.8857500514339197,
    0.8867149807448791,
    0.8850582289925037,
    0.8876335470859426
]
demographic_parity_african_american_reweighted = [0.284916, 0.242086, 0.242086, 0.301676, 0.301676, 0.251397, 0.314711, 0.260708, 0.268156, 0.275605]
demographic_parity_asian_reweighted = [0.312600, 0.274322, 0.269537, 0.277512, 0.341308, 0.290271, 0.315789, 0.285486, 0.318979, 0.291866]
demographic_parity_caucasian_reweighted = [0.233635, 0.210473, 0.205438, 0.244713, 0.273917, 0.216516, 0.279960, 0.227593, 0.263847, 0.238671]
demographic_parity_latin_american_reweighted = [0.241667, 0.229167, 0.200000, 0.262500, 0.287500, 0.220833, 0.291667, 0.241667, 0.279167, 0.245833]
demographic_parity_arab_reweighted = [0.244444, 0.211111, 0.216667, 0.244444, 0.272222, 0.222222, 0.277778, 0.227778, 0.250000, 0.233333]
demographic_parity_unknown_reweighted = [0.244898, 0.193878, 0.208163, 0.248980, 0.240816, 0.210204, 0.259184, 0.218367, 0.238776, 0.222449]

equal_opportunity_african_american_reweighted = [0.804878, 0.743902, 0.719512, 0.792683, 0.792683, 0.743902, 0.804878, 0.731707, 0.756098, 0.756098]
equal_opportunity_asian_reweighted = [0.832117, 0.802920, 0.788321, 0.781022, 0.846715, 0.824818, 0.824818, 0.788321, 0.817518, 0.810219]
equal_opportunity_caucasian_reweighted = [0.672222, 0.638889, 0.616667, 0.694444, 0.755556, 0.633333, 0.777778, 0.677778, 0.744444, 0.688889]
equal_opportunity_latin_american_reweighted = [0.687500, 0.708333, 0.604167, 0.770833, 0.833333, 0.666667, 0.770833, 0.708333, 0.812500, 0.708333]
equal_opportunity_arab_reweighted = [0.607143, 0.571429, 0.571429, 0.607143, 0.607143, 0.571429, 0.678571, 0.571429, 0.607143, 0.571429]
equal_opportunity_unknown_reweighted = [0.739726, 0.684932, 0.712329, 0.753425, 0.753425, 0.657534, 0.780822, 0.726027, 0.739726, 0.712329]

predictive_parity_african_american_reweighted = [0.431373, 0.469231, 0.453846, 0.401235, 0.401235, 0.451852, 0.390533, 0.428571, 0.430556, 0.418919]
predictive_parity_asian_reweighted = [0.581633, 0.639535, 0.639053, 0.614943, 0.542056, 0.620879, 0.570707, 0.603352, 0.560000, 0.606557]
predictive_parity_caucasian_reweighted = [0.521552, 0.550239, 0.544118, 0.514403, 0.500000, 0.530233, 0.503597, 0.539823, 0.511450, 0.523207]
predictive_parity_latin_american_reweighted = [0.568966, 0.618182, 0.604167, 0.587302, 0.579710, 0.603774, 0.528571, 0.586207, 0.582090, 0.576271]
predictive_parity_arab_reweighted = [0.386364, 0.421053, 0.410256, 0.386364, 0.346939, 0.400000, 0.380000, 0.390244, 0.377778, 0.380952]
predictive_parity_unknown_reweighted = [0.450000, 0.526316, 0.509804, 0.450820, 0.466102, 0.466019, 0.448819, 0.495327, 0.461538, 0.477064]



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

for metric_name, groups in fairness_metrics_reweighted.items():
    print(f"\n=== {metric_name} (Reweighted) ===")

    for group, values in groups.items():
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{group}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")

    print(f"\n--- Difference vs Baseline Values ---")
    baseline_groups = fairness_metrics[metric_name]
    for group, reweighted_values in groups.items():
        baseline_values = np.array(baseline_groups[group])
        reweighted_values = np.array(reweighted_values)
        diff = reweighted_values - baseline_values
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        t_stat, p_val = ttest_rel(reweighted_values, baseline_values)
        print(f"{group} - ΔMean: {mean_diff:.4f}, ΔStd: {std_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}")



# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()