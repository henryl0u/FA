import numpy as np
import sys
from scipy.stats import ttest_rel
import json

base_path = "./Fairness/LR/threshold/result"

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

# Accuracy
accuracy_EO = [
    0.817411150961852,
    0.8144766873165961,
    0.8170850994457124,
    0.8089338115422237,
    0.7890446690577111,
    0.8063253994131073,
    0.7505705901532442,
    0.7636126507988262,
    0.7437235083143137,
    0.7795891750896642
]

accuracy_DP = [
    0.8226279752200848,
    0.8092598630583632,
    0.8249103358330616,
    0.8226279752200848,
    0.8226279752200848,
    0.8232800782523638,
    0.8330616237365503,
    0.8252363873492011,
    0.8164329964134334,
    0.8053472448646887
]

# Balanced Accuracy
balanced_accuracy_EO = [
    0.7789015163588842,
    0.786396018000423,
    0.7815586940710455,
    0.7808799112149126,
    0.7559214205613976,
    0.783575483261519,
    0.7624839540658876,
    0.7732792818375963,
    0.7561738814209091,
    0.7737244387907378
]

balanced_accuracy_DP = [
    0.8092062369785253,
    0.8074933425672914,
    0.8041704215842806,
    0.797069642976155,
    0.7984974775646692,
    0.7974666259058889,
    0.7991378660863568,
    0.8022271611663764,
    0.8061488164403091,
    0.7994001066348307
]

# F1 Score
f1_score_EO = [
    0.5845697329376854,
    0.5885755603759942,
    0.5865880619012528,
    0.5796269727403156,
    0.5440451021846371,
    0.57991513437058,
    0.528069093152375,
    0.5437382001258654,
    0.519559902200489,
    0.5535006605019815
]

f1_score_DP = [
    0.6136363636363636,
    0.6012269938650306,
    0.6117136659436009,
    0.604075691411936,
    0.6052249637155298,
    0.6049562682215743,
    0.6150375939849624,
    0.6104651162790697,
    0.6060181945416375,
    0.5919343814080656
]

# EO - Demographic Parity
demographic_parity_african_american_EO = [0.273743, 0.305400, 0.258845, 0.290503, 0.312849, 0.325885, 0.396648, 0.344507, 0.400372, 0.327747]
demographic_parity_asian_EO = [0.263158, 0.285486, 0.282297, 0.293461, 0.267943, 0.283892, 0.387560, 0.338118, 0.334928, 0.301435]
demographic_parity_caucasian_EO = [0.248741, 0.254783, 0.256798, 0.284995, 0.271903, 0.270896, 0.323263, 0.335347, 0.354481, 0.338369]
demographic_parity_latin_american_EO = [0.229167, 0.229167, 0.254167, 0.233333, 0.295833, 0.216667, 0.316667, 0.337500, 0.283333, 0.245833]
demographic_parity_arab_EO = [0.288889, 0.322222, 0.300000, 0.227778, 0.355556, 0.322222, 0.294444, 0.450000, 0.355556, 0.361111]
demographic_parity_unknown_EO = [0.273469, 0.257143, 0.251020, 0.257143, 0.265306, 0.273469, 0.340816, 0.304082, 0.365306, 0.287755]

# DP - Demographic Parity
demographic_parity_african_american_DP = [0.327747, 0.346369, 0.309125, 0.344507, 0.338920, 0.299814, 0.279330, 0.281192, 0.286778, 0.335196]
demographic_parity_asian_DP = [0.263158, 0.259968, 0.285486, 0.258373, 0.275917, 0.283892, 0.264753, 0.248804, 0.303030, 0.269537]
demographic_parity_caucasian_DP = [0.271903, 0.279960, 0.237664, 0.255791, 0.226586, 0.236657, 0.242699, 0.242699, 0.271903, 0.298087]
demographic_parity_latin_american_DP = [0.266667, 0.254167, 0.250000, 0.179167, 0.237500, 0.225000, 0.245833, 0.320833, 0.262500, 0.233333]
demographic_parity_arab_DP = [0.255556, 0.388889, 0.344444, 0.250000, 0.333333, 0.333333, 0.250000, 0.433333, 0.294444, 0.344444]
demographic_parity_unknown_DP = [0.283673, 0.328571, 0.269388, 0.279592, 0.271429, 0.277551, 0.246939, 0.255102, 0.308163, 0.310204]

# EO - Equal Opportunity
equal_opportunity_african_american_EO = [0.731707, 0.768293, 0.707317, 0.768293, 0.695122, 0.804878, 0.841463, 0.853659, 0.768293, 0.780488]
equal_opportunity_asian_EO = [0.751825, 0.751825, 0.744526, 0.781022, 0.766423, 0.766423, 0.781022, 0.773723, 0.759124, 0.766423]
equal_opportunity_caucasian_EO = [0.711111, 0.750000, 0.705556, 0.777778, 0.727778, 0.766667, 0.744444, 0.794444, 0.788889, 0.811111]
equal_opportunity_latin_american_EO = [0.541667, 0.750000, 0.770833, 0.625000, 0.645833, 0.604167, 0.812500, 0.812500, 0.770833, 0.666667]
equal_opportunity_arab_EO = [0.678571, 0.642857, 0.678571, 0.535714, 0.642857, 0.642857, 0.642857, 0.785714, 0.714286, 0.714286]
equal_opportunity_unknown_EO = [0.794521, 0.712329, 0.753425, 0.671233, 0.602740, 0.739726, 0.835616, 0.712329, 0.808219, 0.712329]

# DP - Equal Opportunity
equal_opportunity_african_american_DP = [0.829268, 0.841463, 0.817073, 0.853659, 0.841463, 0.792683, 0.768293, 0.743902, 0.756098, 0.804878]
equal_opportunity_asian_DP = [0.766423, 0.773723, 0.795620, 0.751825, 0.781022, 0.810219, 0.773723, 0.751825, 0.810219, 0.766423]
equal_opportunity_caucasian_DP = [0.794444, 0.805556, 0.727778, 0.755556, 0.694444, 0.705556, 0.716667, 0.733333, 0.794444, 0.811111]
equal_opportunity_latin_american_DP = [0.812500, 0.791667, 0.708333, 0.583333, 0.791667, 0.666667, 0.770833, 0.833333, 0.770833, 0.708333]
equal_opportunity_arab_DP = [0.607143, 0.785714, 0.678571, 0.642857, 0.750000, 0.750000, 0.642857, 0.892857, 0.714286, 0.714286]
equal_opportunity_unknown_DP = [0.821918, 0.849315, 0.821918, 0.821918, 0.780822, 0.808219, 0.767123, 0.808219, 0.821918, 0.821918]

# EO - Predictive Parity
predictive_parity_african_american_EO = [0.408163, 0.384146, 0.417266, 0.403846, 0.339286, 0.377143, 0.323944, 0.378378, 0.293023, 0.363636]
predictive_parity_asian_EO = [0.624242, 0.575419, 0.576271, 0.581522, 0.625000, 0.589888, 0.440329, 0.500000, 0.495238, 0.555556]
predictive_parity_caucasian_EO = [0.518219, 0.533597, 0.498039, 0.494700, 0.485185, 0.513011, 0.417445, 0.429429, 0.403409, 0.434524]
predictive_parity_latin_american_EO = [0.472727, 0.654545, 0.606557, 0.535714, 0.436620, 0.557692, 0.513158, 0.481481, 0.544118, 0.542373]
predictive_parity_arab_EO = [0.365385, 0.310345, 0.351852, 0.365854, 0.281250, 0.310345, 0.339623, 0.271605, 0.312500, 0.307692]
predictive_parity_unknown_EO = [0.432836, 0.412698, 0.447154, 0.388889, 0.338462, 0.402985, 0.365269, 0.348993, 0.329609, 0.368794]

# DP - Predictive Parity
predictive_parity_african_american_DP = [0.386364, 0.365591, 0.403614, 0.378378, 0.379121, 0.403727, 0.420000, 0.403974, 0.402597, 0.366667]
predictive_parity_asian_DP = [0.636364, 0.650307, 0.608939, 0.635802, 0.618497, 0.623596, 0.638554, 0.660256, 0.584211, 0.633136]
predictive_parity_caucasian_DP = [0.529630, 0.521583, 0.555085, 0.535433, 0.555556, 0.540426, 0.535270, 0.547718, 0.529630, 0.493243]
predictive_parity_latin_american_DP = [0.609375, 0.622951, 0.566667, 0.651163, 0.666667, 0.592593, 0.627119, 0.519481, 0.587302, 0.607143]
predictive_parity_arab_DP = [0.369565, 0.314286, 0.354839, 0.400000, 0.350000, 0.350000, 0.400000, 0.320513, 0.377358, 0.322581]
predictive_parity_unknown_DP = [0.431655, 0.385093, 0.454545, 0.437956, 0.428571, 0.433824, 0.462810, 0.472000, 0.397351, 0.394737]




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

reference_group = "Caucasian"

for metric_name, groups in fairness_metrics_DP.items():
    print(f"\n=== {metric_name} (TO-DP) ===")
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

for metric_name, groups in fairness_metrics_EO.items():
    print(f"\n=== {metric_name} (TO-EO) ===")
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
with open("./Fairness/LR/base/result/fairness_differences.json") as f:
    baseline_diffs = json.load(f)

with open("./Fairness/LR/threshold/result/fairness_differences_DP.json") as f:
    mitigated_diffs_DP = json.load(f)

with open("./Fairness/LR/threshold/result/fairness_differences_EO.json") as f:
    mitigated_diffs_EO = json.load(f)

# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== Disparity Comparison DP: {metric_name} ===")
    
    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs_DP[metric_name][group])
        
        t_stat, p_val = ttest_rel(mitigated_diff, base_diff)

        print(f"{group} - t={t_stat:.3f}, p={p_val:.4f}")

# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== Disparity Comparison EO: {metric_name} ===")
    
    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs_EO[metric_name][group])
        
        t_stat, p_val = ttest_rel(mitigated_diff, base_diff)

        print(f"{group} - t={t_stat:.3f}, p={p_val:.4f}")

# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== Disparity Comparison DP (ABS): {metric_name} ===")
    
    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs_DP[metric_name][group])
        
        t_stat, p_val = ttest_rel(np.abs(mitigated_diff), np.abs(base_diff))

        print(f"{group} - t={t_stat:.3f}, p={p_val:.4f}")

# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== Disparity Comparison EO (ABS): {metric_name} ===")
    
    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs_EO[metric_name][group])
        
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
