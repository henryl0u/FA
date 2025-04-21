import numpy as np
import sys
from scipy.stats import ttest_rel

base_path = "./Fairness/SVM/threshold/result"

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


# Accuracy
accuracy_DP = [
    0.8278447994783176,
    0.8268666449298989,
    0.8167590479295729,
    0.8281708509944571,
    0.8086077600260841,
    0.8268666449298989,
    0.8294750570590154,
    0.8415389631561787,
    0.8190414085425497,
    0.8144766873165961
]

accuracy_EO = [
    0.8245842843169221,
    0.7750244538637104,
    0.8281708509944571,
    0.8134985327681774,
    0.7544832083469188,
    0.8040430388001304,
    0.752852950766221,
    0.7910009781545484,
    0.7763286599282687,
    0.7936093902836648
]

# Balanced Accuracy
balanced_accuracy_DP = [
    0.8109542658278832,
    0.8003639493136832,
    0.7927828793142917,
    0.8004439978788942,
    0.8013850212835009,
    0.7875134380170558,
    0.7955266253843056,
    0.8071543133499275,
    0.8013114925109315,
    0.8085274541223924
]

balanced_accuracy_EO = [
    0.7754152383491305,
    0.78308215228497,
    0.7797403963454389,
    0.7865144609000791,
    0.7634380170557775,
    0.7850417121844783,
    0.7717264845567845,
    0.7792438054725691,
    0.7660281857880111,
    0.780117819897248
]

# F1 Score
f1_score_DP = [
    0.6195965417867435,
    0.6104181951577403,
    0.5956834532374101,
    0.6116433308769345,
    0.5960082587749483,
    0.5998492840994725,
    0.6088257292445775,
    0.6295731707317073,
    0.6044191019244476,
    0.6062283737024221
]

f1_score_EO = [
    0.5874233128834356,
    0.558258642765685,
    0.5943033102386451,
    0.5878962536023055,
    0.5308411214953271,
    0.5794261721483555,
    0.5366748166259169,
    0.5654237288135593,
    0.5450928381962865,
    0.5679180887372014
]

# Demographic Parity (DP)
demographic_parity_african_american_DP = [0.329609, 0.331471, 0.324022, 0.335196, 0.357542, 0.270019, 0.279330, 0.229050, 0.277467, 0.320298]
demographic_parity_asian_DP = [0.259968, 0.237640, 0.290271, 0.256778, 0.296651, 0.287081, 0.266348, 0.244019, 0.303030, 0.264753]
demographic_parity_caucasian_DP = [0.269889, 0.238671, 0.224572, 0.235650, 0.259819, 0.215509, 0.240685, 0.229607, 0.258812, 0.290030]
demographic_parity_latin_american_DP = [0.229167, 0.225000, 0.254167, 0.195833, 0.258333, 0.208333, 0.258333, 0.316667, 0.258333, 0.233333]
demographic_parity_arab_DP = [0.244444, 0.338889, 0.394444, 0.283333, 0.372222, 0.350000, 0.272222, 0.350000, 0.272222, 0.338889]
demographic_parity_unknown_DP = [0.271429, 0.277551, 0.267347, 0.277551, 0.285714, 0.259184, 0.248980, 0.246939, 0.302041, 0.314286]

# Equal Opportunity (DP)
equal_opportunity_african_american_DP = [0.853659, 0.804878, 0.817073, 0.841463, 0.829268, 0.731707, 0.756098, 0.719512, 0.780488, 0.817073]
equal_opportunity_asian_DP = [0.781022, 0.744526, 0.802920, 0.759124, 0.802920, 0.802920, 0.773723, 0.729927, 0.802920, 0.781022]
equal_opportunity_caucasian_DP = [0.777778, 0.738889, 0.683333, 0.733333, 0.766667, 0.672222, 0.722222, 0.722222, 0.761111, 0.805556]
equal_opportunity_latin_american_DP = [0.729167, 0.666667, 0.750000, 0.625000, 0.750000, 0.625000, 0.770833, 0.916667, 0.708333, 0.729167]
equal_opportunity_arab_DP = [0.642857, 0.785714, 0.714286, 0.714286, 0.785714, 0.750000, 0.642857, 0.857143, 0.678571, 0.750000]
equal_opportunity_unknown_DP = [0.821918, 0.835616, 0.794521, 0.821918, 0.808219, 0.767123, 0.739726, 0.767123, 0.821918, 0.863014]

# Predictive Parity (DP)
predictive_parity_african_american_DP = [0.395480, 0.370787, 0.385057, 0.383333, 0.354167, 0.413793, 0.413333, 0.479675, 0.429530, 0.389535]
predictive_parity_asian_DP = [0.656442, 0.684564, 0.604396, 0.645963, 0.591398, 0.611111, 0.634731, 0.653595, 0.578947, 0.644578]
predictive_parity_caucasian_DP = [0.522388, 0.561181, 0.551570, 0.564103, 0.534884, 0.565421, 0.543933, 0.570175, 0.533074, 0.503472]
predictive_parity_latin_american_DP = [0.636364, 0.592593, 0.590164, 0.638298, 0.580645, 0.600000, 0.596774, 0.578947, 0.548387, 0.625000]
predictive_parity_arab_DP = [0.409091, 0.360656, 0.281690, 0.392157, 0.328358, 0.333333, 0.367347, 0.380952, 0.387755, 0.344262]
predictive_parity_unknown_DP = [0.451128, 0.448529, 0.442748, 0.441176, 0.421429, 0.440945, 0.442623, 0.462810, 0.405405, 0.409091]

# Demographic Parity (EO)
demographic_parity_african_american_EO = [0.247672, 0.379888, 0.255121, 0.312849, 0.398510, 0.305400, 0.396648, 0.294227, 0.329609, 0.329609]
demographic_parity_asian_EO = [0.267943, 0.336523, 0.267943, 0.283892, 0.295056, 0.299841, 0.408293, 0.299841, 0.336523, 0.277512]
demographic_parity_caucasian_EO = [0.240685, 0.319235, 0.228600, 0.274924, 0.347432, 0.264854, 0.329305, 0.303122, 0.290030, 0.313192]
demographic_parity_latin_american_EO = [0.233333, 0.241667, 0.237500, 0.191667, 0.312500, 0.245833, 0.270833, 0.329167, 0.279167, 0.220833]
demographic_parity_arab_EO = [0.200000, 0.366667, 0.255556, 0.272222, 0.405556, 0.366667, 0.355556, 0.388889, 0.311111, 0.338889]
demographic_parity_unknown_EO = [0.253061, 0.322449, 0.236735, 0.257143, 0.336735, 0.287755, 0.332653, 0.267347, 0.328571, 0.287755]

# Equal Opportunity (EO)
equal_opportunity_african_american_EO = [0.695122, 0.841463, 0.743902, 0.817073, 0.878049, 0.756098, 0.853659, 0.780488, 0.658537, 0.792683]
equal_opportunity_asian_EO = [0.759124, 0.810219, 0.722628, 0.744526, 0.737226, 0.802920, 0.846715, 0.744526, 0.795620, 0.729927]
equal_opportunity_caucasian_EO = [0.666667, 0.805556, 0.666667, 0.777778, 0.805556, 0.722222, 0.772222, 0.744444, 0.738889, 0.788889]
equal_opportunity_latin_american_EO = [0.645833, 0.729167, 0.750000, 0.541667, 0.708333, 0.750000, 0.770833, 0.875000, 0.833333, 0.645833]
equal_opportunity_arab_EO = [0.428571, 0.642857, 0.607143, 0.678571, 0.714286, 0.714286, 0.642857, 0.821429, 0.535714, 0.785714]
equal_opportunity_unknown_EO = [0.808219, 0.794521, 0.726027, 0.739726, 0.739726, 0.767123, 0.808219, 0.712329, 0.821918, 0.767123]

# Predictive Parity (EO)
predictive_parity_african_american_EO = [0.428571, 0.338235, 0.445255, 0.398810, 0.336449, 0.378049, 0.328638, 0.405063, 0.305085, 0.367232]
predictive_parity_asian_EO = [0.619048, 0.526066, 0.589286, 0.573034, 0.545946, 0.585106, 0.453125, 0.542553, 0.516588, 0.574713]
predictive_parity_caucasian_EO = [0.502092, 0.457413, 0.528634, 0.512821, 0.420290, 0.494297, 0.425076, 0.445183, 0.461806, 0.456592]
predictive_parity_latin_american_EO = [0.553571, 0.603448, 0.631579, 0.565217, 0.453333, 0.610169, 0.569231, 0.531646, 0.597015, 0.584906]
predictive_parity_arab_EO = [0.333333, 0.272727, 0.369565, 0.387755, 0.273973, 0.303030, 0.281250, 0.328571, 0.267857, 0.360656]
predictive_parity_unknown_EO = [0.475806, 0.367089, 0.456897, 0.428571, 0.327273, 0.397163, 0.361963, 0.396947, 0.372671, 0.397163]



def print_perf_stats(metric_name, values):
    print(f"{metric_name} - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")


# Print performance stats for egr metrics
print("\n=== TO Performance Metrics ===")
print_perf_stats("Accuracy (EO)", accuracy_EO)
print_perf_stats("Balanced Accuracy (EO)", balanced_accuracy_EO)
print_perf_stats("F1 Score (EO)", f1_score_EO)
print_perf_stats("Accuracy (DP)", accuracy_DP)
print_perf_stats("Balanced Accuracy (DP)", balanced_accuracy_DP)
print_perf_stats("F1 Score (DP)", f1_score_DP)


def compare_and_print(metric_name, baseline_values, egr_values):
    baseline_values = np.array(baseline_values)
    egr_values = np.array(egr_values)
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
compare_and_print("Accuracy (DP)", accuracy, accuracy_DP)
compare_and_print("Balanced Accuracy (DP)", balanced_accuracy, balanced_accuracy_DP)
compare_and_print("F1 Score (DP)", f1_score, f1_score_DP)

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

for metric_name, groups in fairness_metrics_EO.items():
    print(f"\n=== {metric_name} (EO) ===")

    for group, values in groups.items():
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{group}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")

    print(f"\n--- Difference vs Baseline Values ---")
    baseline_groups = fairness_metrics[metric_name]
    for group, egr_values in groups.items():
        baseline_values = np.array(baseline_groups[group])
        egr_values = np.array(egr_values)
        diff = egr_values - baseline_values
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        t_stat, p_val = ttest_rel(egr_values, baseline_values)
        print(
            f"{group} - ΔMean: {mean_diff:.4f}, ΔStd: {std_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}"
        )

for metric_name, groups in fairness_metrics_DP.items():
    print(f"\n=== {metric_name} (DP) ===")

    for group, values in groups.items():
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{group}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")

    print(f"\n--- Difference vs Baseline Values ---")
    baseline_groups = fairness_metrics[metric_name]
    for group, egr_values in groups.items():
        baseline_values = np.array(baseline_groups[group])
        egr_values = np.array(egr_values)
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
