import numpy as np
import sys
from scipy.stats import ttest_rel

base_path = "./Fairness/RF/egr/result"

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
demographic_parity_african_american_EO = [0.305400, 0.279330, 0.275605, 0.266294, 0.283054]
demographic_parity_asian_EO = [0.323764, 0.303030, 0.301435, 0.282297, 0.303030]
demographic_parity_caucasian_EO = [0.304129, 0.287009, 0.278953, 0.265861, 0.293051]
demographic_parity_latin_american_EO = [0.308333, 0.275000, 0.283333, 0.241667, 0.270833]
demographic_parity_arab_EO = [0.300000, 0.277778, 0.277778, 0.261111, 0.272222]
demographic_parity_unknown_EO = [0.261224, 0.251020, 0.255102, 0.242857, 0.238776]

# DP
demographic_parity_african_american_DP = [0.331471, 0.335196, 0.264432, 0.232775, 0.245810]
demographic_parity_asian_DP = [0.306220, 0.314195, 0.259968, 0.236045, 0.237640]
demographic_parity_caucasian_DP = [0.292044, 0.283988, 0.241692, 0.229607, 0.224572]
demographic_parity_latin_american_DP = [0.304167, 0.295833, 0.237500, 0.212500, 0.225000]
demographic_parity_arab_DP = [0.388889, 0.377778, 0.272222, 0.261111, 0.233333]
demographic_parity_unknown_DP = [0.314286, 0.359184, 0.263265, 0.246939, 0.240816]


# EO
equal_opportunity_african_american_EO = [0.817073, 0.768293, 0.743902, 0.743902, 0.792683]
equal_opportunity_asian_EO = [0.824818, 0.824818, 0.810219, 0.795620, 0.810219]
equal_opportunity_caucasian_EO = [0.822222, 0.805556, 0.811111, 0.805556, 0.816667]
equal_opportunity_latin_american_EO = [0.895833, 0.833333, 0.833333, 0.770833, 0.791667]
equal_opportunity_arab_EO = [0.714286, 0.678571, 0.678571, 0.607143, 0.678571]
equal_opportunity_unknown_EO = [0.821918, 0.821918, 0.794521, 0.780822, 0.794521]

# DP
equal_opportunity_african_american_DP = [0.841463, 0.829268, 0.743902, 0.707317, 0.707317]
equal_opportunity_asian_DP = [0.810219, 0.824818, 0.759124, 0.729927, 0.744526]
equal_opportunity_caucasian_DP = [0.816667, 0.816667, 0.761111, 0.722222, 0.722222]
equal_opportunity_latin_american_DP = [0.875000, 0.854167, 0.770833, 0.666667, 0.708333]
equal_opportunity_arab_DP = [0.892857, 0.857143, 0.678571, 0.642857, 0.571429]
equal_opportunity_unknown_DP = [0.849315, 0.904110, 0.821918, 0.794521, 0.780822]


# EO
predictive_parity_african_american_EO = [0.408537, 0.420000, 0.412162, 0.426573, 0.427632]
predictive_parity_asian_EO = [0.556650, 0.594737, 0.587302, 0.615819, 0.584211]
predictive_parity_caucasian_EO = [0.490066, 0.508772, 0.527076, 0.549242, 0.505155]
predictive_parity_latin_american_EO = [0.581081, 0.606061, 0.588235, 0.637931, 0.584615]
predictive_parity_arab_EO = [0.370370, 0.380000, 0.380000, 0.361702, 0.387755]
predictive_parity_unknown_EO = [0.468750, 0.487805, 0.464000, 0.478992, 0.495726]

# DP
predictive_parity_african_american_DP = [0.387640, 0.377778, 0.429577, 0.464000, 0.439394]
predictive_parity_asian_DP = [0.578125, 0.573604, 0.638037, 0.675676, 0.684564]
predictive_parity_caucasian_DP = [0.506897, 0.521277, 0.570833, 0.570175, 0.582960]
predictive_parity_latin_american_DP = [0.575342, 0.577465, 0.649123, 0.627451, 0.629630]
predictive_parity_arab_DP = [0.357143, 0.352941, 0.387755, 0.382979, 0.380952]
predictive_parity_unknown_DP = [0.402597, 0.375000, 0.465116, 0.479339, 0.483051]



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
