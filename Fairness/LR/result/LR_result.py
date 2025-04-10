import numpy as np
import sys
from scipy.stats import ttest_rel

base_path = "./Fairness/LR/result"

# Save the default standard output
default_stdout = sys.stdout

# Open the file you want to redirect output to
f = open(f"{base_path}/prints.txt", "w")

# Change standard output to point to the file
sys.stdout = f

accuracy = [
    0.8161069448972937,
    0.8258884903814803,
    0.8373002934463645,
    0.8229540267362243,
    0.7962178024127812,
    0.8288229540267362,
    0.8236061297685034,
    0.8268666449298989,
    0.8232800782523638,
    0.8252363873492011
]

balanced_accuracy = [
    0.8080920768582134,
    0.8154746553927378,
    0.792437330304286,
    0.806549059266364,
    0.8116902779749814,
    0.799413146220114,
    0.8026625384305555,
    0.8025057011964544,
    0.8096032199082592,
    0.8050828303434048
]

f1_score = [
    0.6072423398328691,
    0.6212765957446809,
    0.6134779240898528,
    0.6118656182987848,
    0.5944192083062946,
    0.6113989637305699,
    0.6093862815884477,
    0.6121256391526662,
    0.6145092460881935,
    0.6127167630057804
]

roc_auc_score = [
    0.8874408509923124,
    0.8875871841160464,
    0.8860151896680123,
    0.8861093644506133,
    0.8860072210325612,
    0.8861847042766942,
    0.8853088787985037,
    0.8858775496011336,
    0.883953486350452,
    0.8868040845776478
]

demographic_parity_african_american = [0.322160, 0.297952, 0.256983, 0.312849, 0.337058, 0.348231, 0.294227, 0.266294, 0.275605, 0.292365]
demographic_parity_asian = [0.333333, 0.315789, 0.282297, 0.291866, 0.360447, 0.366826, 0.298246, 0.282297, 0.320574, 0.301435]
demographic_parity_caucasian = [0.278953, 0.282981, 0.241692, 0.272910, 0.343404, 0.331319, 0.272910, 0.242699, 0.277946, 0.279960]
demographic_parity_latin_american = [0.241667, 0.262500, 0.212500, 0.241667, 0.300000, 0.295833, 0.250000, 0.233333, 0.270833, 0.245833]
demographic_parity_arab = [0.244444, 0.250000, 0.200000, 0.250000, 0.261111, 0.300000, 0.244444, 0.211111, 0.255556, 0.222222]
demographic_parity_unknown = [0.259184, 0.234694, 0.206122, 0.257143, 0.257143, 0.273469, 0.238776, 0.210204, 0.248980, 0.230612]

equal_opportunity_african_american = [0.853659, 0.817073, 0.707317, 0.804878, 0.853659, 0.853659, 0.780488, 0.719512, 0.743902, 0.756098]
equal_opportunity_asian = [0.832117, 0.846715, 0.795620, 0.795620, 0.868613, 0.875912, 0.817518, 0.788321, 0.832117, 0.817518]
equal_opportunity_caucasian = [0.788889, 0.794444, 0.727778, 0.783333, 0.850000, 0.838889, 0.777778, 0.733333, 0.788889, 0.788889]
equal_opportunity_latin_american = [0.729167, 0.812500, 0.645833, 0.750000, 0.854167, 0.791667, 0.729167, 0.708333, 0.833333, 0.770833]
equal_opportunity_arab = [0.642857, 0.642857, 0.500000, 0.642857, 0.607143, 0.714286, 0.607143, 0.571429, 0.642857, 0.607143]
equal_opportunity_unknown = [0.780822, 0.753425, 0.726027, 0.794521, 0.794521, 0.808219, 0.739726, 0.712329, 0.780822, 0.739726]

predictive_parity_african_american = [0.404624, 0.418750, 0.420290, 0.392857, 0.386740, 0.374332, 0.405063, 0.412587, 0.412162, 0.394904]
predictive_parity_asian = [0.545455, 0.585859, 0.615819, 0.595628, 0.526549, 0.521739, 0.598930, 0.610169, 0.567164, 0.592593]
predictive_parity_caucasian = [0.512635, 0.508897, 0.545833, 0.520295, 0.448680, 0.458967, 0.516605, 0.547718, 0.514493, 0.510791]
predictive_parity_latin_american = [0.603448, 0.619048, 0.607843, 0.620690, 0.569444, 0.535211, 0.583333, 0.607143, 0.615385, 0.627119]
predictive_parity_arab = [0.409091, 0.400000, 0.388889, 0.400000, 0.361702, 0.370370, 0.386364, 0.421053, 0.391304, 0.425000]
predictive_parity_unknown = [0.448819, 0.478261, 0.524752, 0.460317, 0.460317, 0.440299, 0.461538, 0.504854, 0.467213, 0.477876]

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

# Demographic Parity (Reweighted)
demographic_parity_african_american_reweighted = [0.284916, 0.242086, 0.242086, 0.301676, 0.301676, 0.251397, 0.314711, 0.260708, 0.268156, 0.275605]
demographic_parity_asian_reweighted = [0.312600, 0.274322, 0.269537, 0.277512, 0.341308, 0.290271, 0.315789, 0.285486, 0.318979, 0.291866]
demographic_parity_caucasian_reweighted = [0.233635, 0.210473, 0.205438, 0.244713, 0.273917, 0.216516, 0.279960, 0.227593, 0.263847, 0.238671]
demographic_parity_latin_american_reweighted = [0.241667, 0.229167, 0.200000, 0.262500, 0.287500, 0.220833, 0.291667, 0.241667, 0.279167, 0.245833]
demographic_parity_arab_reweighted = [0.244444, 0.211111, 0.216667, 0.244444, 0.272222, 0.222222, 0.277778, 0.227778, 0.250000, 0.233333]
demographic_parity_unknown_reweighted = [0.244898, 0.193878, 0.208163, 0.248980, 0.240816, 0.210204, 0.259184, 0.218367, 0.238776, 0.222449]

# Equal Opportunity (Reweighted)
equal_opportunity_african_american_reweighted = [0.804878, 0.743902, 0.719512, 0.792683, 0.792683, 0.743902, 0.804878, 0.731707, 0.756098, 0.756098]
equal_opportunity_asian_reweighted = [0.832117, 0.802920, 0.788321, 0.781022, 0.846715, 0.824818, 0.824818, 0.788321, 0.817518, 0.810219]
equal_opportunity_caucasian_reweighted = [0.672222, 0.638889, 0.616667, 0.694444, 0.755556, 0.633333, 0.777778, 0.677778, 0.744444, 0.688889]
equal_opportunity_latin_american_reweighted = [0.687500, 0.708333, 0.604167, 0.770833, 0.833333, 0.666667, 0.770833, 0.708333, 0.812500, 0.708333]
equal_opportunity_arab_reweighted = [0.607143, 0.571429, 0.571429, 0.607143, 0.607143, 0.571429, 0.678571, 0.571429, 0.607143, 0.571429]
equal_opportunity_unknown_reweighted = [0.739726, 0.684932, 0.712329, 0.753425, 0.753425, 0.657534, 0.780822, 0.726027, 0.739726, 0.712329]

# Predictive Parity (Reweighted)
predictive_parity_african_american_reweighted = [0.431373, 0.469231, 0.453846, 0.401235, 0.401235, 0.451852, 0.390533, 0.428571, 0.430556, 0.418919]
predictive_parity_asian_reweighted = [0.581633, 0.639535, 0.639053, 0.614943, 0.542056, 0.620879, 0.570707, 0.603352, 0.560000, 0.606557]
predictive_parity_caucasian_reweighted = [0.521552, 0.550239, 0.544118, 0.514403, 0.500000, 0.530233, 0.503597, 0.539823, 0.511450, 0.523207]
predictive_parity_latin_american_reweighted = [0.568966, 0.618182, 0.604167, 0.587302, 0.579710, 0.603774, 0.528571, 0.586207, 0.582090, 0.576271]
predictive_parity_arab_reweighted = [0.386364, 0.421053, 0.410256, 0.386364, 0.346939, 0.400000, 0.380000, 0.390244, 0.377778, 0.380952]
predictive_parity_unknown_reweighted = [0.450000, 0.526316, 0.509804, 0.450820, 0.466102, 0.466019, 0.448819, 0.495327, 0.461538, 0.477064]




def print_perf_stats(metric_name, values):
    print(f"{metric_name} - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")

print("\n=== Performance Metrics ===")
print_perf_stats("Accuracy", accuracy)
print_perf_stats("Balanced Accuracy", balanced_accuracy)
print_perf_stats("F1 Score", f1_score)
print_perf_stats("ROC AUC Score", roc_auc_score)

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