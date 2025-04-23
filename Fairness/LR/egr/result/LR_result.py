import numpy as np
import sys
from scipy.stats import ttest_rel

base_path = "./Fairness/LR/egr/result"

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
    0.8252363873492011,
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
    0.8050828303434048,
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
    0.6127167630057804,
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
    0.8868040845776478,
]

demographic_parity_african_american = [
    0.322160,
    0.297952,
    0.256983,
    0.312849,
    0.337058,
    0.348231,
    0.294227,
    0.266294,
    0.275605,
    0.292365,
]
demographic_parity_asian = [
    0.333333,
    0.315789,
    0.282297,
    0.291866,
    0.360447,
    0.366826,
    0.298246,
    0.282297,
    0.320574,
    0.301435,
]
demographic_parity_caucasian = [
    0.278953,
    0.282981,
    0.241692,
    0.272910,
    0.343404,
    0.331319,
    0.272910,
    0.242699,
    0.277946,
    0.279960,
]
demographic_parity_latin_american = [
    0.241667,
    0.262500,
    0.212500,
    0.241667,
    0.300000,
    0.295833,
    0.250000,
    0.233333,
    0.270833,
    0.245833,
]
demographic_parity_arab = [
    0.244444,
    0.250000,
    0.200000,
    0.250000,
    0.261111,
    0.300000,
    0.244444,
    0.211111,
    0.255556,
    0.222222,
]
demographic_parity_unknown = [
    0.259184,
    0.234694,
    0.206122,
    0.257143,
    0.257143,
    0.273469,
    0.238776,
    0.210204,
    0.248980,
    0.230612,
]

equal_opportunity_african_american = [
    0.853659,
    0.817073,
    0.707317,
    0.804878,
    0.853659,
    0.853659,
    0.780488,
    0.719512,
    0.743902,
    0.756098,
]
equal_opportunity_asian = [
    0.832117,
    0.846715,
    0.795620,
    0.795620,
    0.868613,
    0.875912,
    0.817518,
    0.788321,
    0.832117,
    0.817518,
]
equal_opportunity_caucasian = [
    0.788889,
    0.794444,
    0.727778,
    0.783333,
    0.850000,
    0.838889,
    0.777778,
    0.733333,
    0.788889,
    0.788889,
]
equal_opportunity_latin_american = [
    0.729167,
    0.812500,
    0.645833,
    0.750000,
    0.854167,
    0.791667,
    0.729167,
    0.708333,
    0.833333,
    0.770833,
]
equal_opportunity_arab = [
    0.642857,
    0.642857,
    0.500000,
    0.642857,
    0.607143,
    0.714286,
    0.607143,
    0.571429,
    0.642857,
    0.607143,
]
equal_opportunity_unknown = [
    0.780822,
    0.753425,
    0.726027,
    0.794521,
    0.794521,
    0.808219,
    0.739726,
    0.712329,
    0.780822,
    0.739726,
]

predictive_parity_african_american = [
    0.404624,
    0.418750,
    0.420290,
    0.392857,
    0.386740,
    0.374332,
    0.405063,
    0.412587,
    0.412162,
    0.394904,
]
predictive_parity_asian = [
    0.545455,
    0.585859,
    0.615819,
    0.595628,
    0.526549,
    0.521739,
    0.598930,
    0.610169,
    0.567164,
    0.592593,
]
predictive_parity_caucasian = [
    0.512635,
    0.508897,
    0.545833,
    0.520295,
    0.448680,
    0.458967,
    0.516605,
    0.547718,
    0.514493,
    0.510791,
]
predictive_parity_latin_american = [
    0.603448,
    0.619048,
    0.607843,
    0.620690,
    0.569444,
    0.535211,
    0.583333,
    0.607143,
    0.615385,
    0.627119,
]
predictive_parity_arab = [
    0.409091,
    0.400000,
    0.388889,
    0.400000,
    0.361702,
    0.370370,
    0.386364,
    0.421053,
    0.391304,
    0.425000,
]
predictive_parity_unknown = [
    0.448819,
    0.478261,
    0.524752,
    0.460317,
    0.460317,
    0.440299,
    0.461538,
    0.504854,
    0.467213,
    0.477876,
]

# EGR Model Metrics (New Batch)

# Accuracy
accuracy_EO = [
    0.8278447994783176,
    0.8386044995109228,
    0.8275187479621781,
    0.8252363873492011,
    0.8200195630909684,
]

accuracy_DP = [
    0.8412129116400391,
    0.8402347570916204,
    0.8255624388653408,
    0.8268666449298989,
    0.8144766873165961,
]

# Balanced Accuracy
balanced_accuracy_EO = [
    0.800245506414027,
    0.8103653112259238,
    0.8036166014204454,
    0.815791589757261,
    0.8104739744366175,
]

balanced_accuracy_DP = [
    0.8098114910620887,
    0.8170691069043156,
    0.8152761639278707,
    0.8160701297873388,
    0.8149527097707061,
]

# F1 Score
f1_score_EO = [
    0.6111929307805597,
    0.6292134831460674,
    0.6135865595325055,
    0.620933521923621,
    0.6123595505617978,
]

f1_score_DP = [
    0.631339894019682,
    0.6359583952451708,
    0.6208362863217576,
    0.6226012793176973,
    0.6110731373889269,
]

# ROC AUC
roc_auc_EO = [
    0.8881580281828905,
    0.892306065145768,
    0.8900002318148494,
    0.8935506211189124,
    0.8911803142829824,
]

roc_auc_DP = [
    0.8932239070654268,
    0.8913085368716007,
    0.8904942872128031,
    0.8902124872864045,
    0.8916229357612075,
]


# Demographic Parity – EO
demographic_parity_african_american_EO = [
    0.288641,
    0.260708,
    0.288641,
    0.290503,
    0.275605,
]
demographic_parity_asian_EO = [0.290271, 0.275917, 0.279107, 0.299841, 0.311005]
demographic_parity_caucasian_EO = [0.249748, 0.246727, 0.258812, 0.274924, 0.290030]
demographic_parity_latin_american_EO = [
    0.233333,
    0.233333,
    0.241667,
    0.241667,
    0.258333,
]
demographic_parity_arab_EO = [0.250000, 0.250000, 0.250000, 0.294444, 0.277778]
demographic_parity_unknown_EO = [0.253061, 0.261224, 0.267347, 0.281633, 0.271429]

# Demographic Parity – DP
demographic_parity_african_american_DP = [
    0.255121,
    0.270019,
    0.305400,
    0.283054,
    0.303538,
]
demographic_parity_asian_DP = [0.269537, 0.266348, 0.290271, 0.285486, 0.296651]
demographic_parity_caucasian_DP = [0.234642, 0.237664, 0.261833, 0.258812, 0.288016]
demographic_parity_latin_american_DP = [
    0.233333,
    0.241667,
    0.241667,
    0.237500,
    0.262500,
]
demographic_parity_arab_DP = [0.294444, 0.277778, 0.300000, 0.361111, 0.333333]
demographic_parity_unknown_DP = [0.255102, 0.289796, 0.295918, 0.304082, 0.320408]

# Equal Opportunity – EO
equal_opportunity_african_american_EO = [
    0.768293,
    0.780488,
    0.792683,
    0.780488,
    0.756098,
]
equal_opportunity_asian_EO = [0.766423, 0.795620, 0.802920, 0.817518, 0.810219]
equal_opportunity_caucasian_EO = [0.761111, 0.761111, 0.738889, 0.805556, 0.816667]
equal_opportunity_latin_american_EO = [0.729167, 0.750000, 0.770833, 0.791667, 0.791667]
equal_opportunity_arab_EO = [0.607143, 0.607143, 0.607143, 0.678571, 0.642857]
equal_opportunity_unknown_EO = [0.794521, 0.780822, 0.794521, 0.835616, 0.821918]

# Equal Opportunity – DP
equal_opportunity_african_american_DP = [
    0.743902,
    0.792683,
    0.817073,
    0.804878,
    0.829268,
]
equal_opportunity_asian_DP = [0.802920, 0.795620, 0.817518, 0.795620, 0.810219]
equal_opportunity_caucasian_DP = [0.733333, 0.750000, 0.772222, 0.783333, 0.816667]
equal_opportunity_latin_american_DP = [0.750000, 0.791667, 0.812500, 0.791667, 0.812500]
equal_opportunity_arab_DP = [0.678571, 0.678571, 0.678571, 0.785714, 0.714286]
equal_opportunity_unknown_DP = [0.808219, 0.849315, 0.849315, 0.849315, 0.849315]

# Predictive Parity – EO
predictive_parity_african_american_EO = [
    0.406452,
    0.457143,
    0.419355,
    0.410256,
    0.418919,
]
predictive_parity_asian_EO = [0.576923, 0.630058, 0.628571, 0.595745, 0.569231]
predictive_parity_caucasian_EO = [0.552419, 0.559184, 0.517510, 0.531136, 0.510417]
predictive_parity_latin_american_EO = [0.625000, 0.642857, 0.637931, 0.655172, 0.612903]
predictive_parity_arab_EO = [0.377778, 0.377778, 0.377778, 0.358491, 0.360000]
predictive_parity_unknown_EO = [0.467742, 0.445312, 0.442748, 0.442029, 0.451128]

# Predictive Parity – DP
predictive_parity_african_american_DP = [
    0.445255,
    0.448276,
    0.408537,
    0.434211,
    0.417178,
]
predictive_parity_asian_DP = [0.650888, 0.652695, 0.615385, 0.608939, 0.596774]
predictive_parity_caucasian_DP = [0.566524, 0.572034, 0.534615, 0.548638, 0.513986]
predictive_parity_latin_american_DP = [0.642857, 0.655172, 0.672414, 0.666667, 0.619048]
predictive_parity_arab_DP = [0.358491, 0.380000, 0.351852, 0.338462, 0.333333]
predictive_parity_unknown_DP = [0.472000, 0.436620, 0.427586, 0.416107, 0.394904]


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




# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()
