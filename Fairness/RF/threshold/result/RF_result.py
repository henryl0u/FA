import numpy as np
import sys
from scipy.stats import ttest_rel

base_path = "./Fairness/RF/threshold/result"

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

# TO Model Metrics (New Batch)

# Accuracy
accuracy_EO = [
    0.8148027388327356,
    0.7880665145092924,
    0.831105314639713,
    0.8298011085751549,
    0.7548092598630584,
    0.8219758721878057,
    0.7962178024127812,
    0.7727420932507336,
    0.8076296054776655,
    0.817411150961852
]

accuracy_DP = [
    0.8177372024779915,
    0.817411150961852,
    0.8327355722204108,
    0.8275187479621781,
    0.8115422236713401,
    0.8386044995109228,
    0.8320834691881317,
    0.8425171177045974,
    0.8245842843169221,
    0.834039778284969
]

# Balanced Accuracy
balanced_accuracy_EO = [
    0.7801692538169763,
    0.7760295477002518,
    0.7372639472853033,
    0.780732853669774,
    0.7586390874608451,
    0.7552654569795104,
    0.7824196689104412,
    0.7559916894376462,
    0.7493875017023903,
    0.7917520276555114
]

balanced_accuracy_DP = [
    0.8090845341825483,
    0.8103138773061955,
    0.7996532919157469,
    0.8071861878917309,
    0.8045992790558181,
    0.7910895442809828,
    0.799256308986013,
    0.7891879380938445,
    0.8118250203562414,
    0.801875092363729
]

# F1 Score
f1_score_EO = [
    0.5835777126099707,
    0.5608108108108109,
    0.5557461406518011,
    0.5965996908809892,
    0.5270440251572327,
    0.5666666666666667,
    0.5716244002741604,
    0.5344021376085505,
    0.5503048780487805,
    0.5953757225433526
]

f1_score_DP = [
    0.6093640810621943,
    0.6100278551532033,
    0.6151537884471118,
    0.6163886874546773,
    0.600828729281768,
    0.6135831381733021,
    0.6142322097378277,
    0.6157517899761337,
    0.6173541963015647,
    0.6181545386346586
]

# EO - Demographic Parity
demographic_parity_african_american_EO = [0.260708, 0.346369, 0.206704, 0.247672, 0.374302, 0.258845, 0.305400, 0.271881, 0.245810, 0.288641]
demographic_parity_asian_EO = [0.261563, 0.309410, 0.213716, 0.240829, 0.298246, 0.245614, 0.352472, 0.309410, 0.245614, 0.264753]
demographic_parity_caucasian_EO = [0.280967, 0.293051, 0.180262, 0.258812, 0.340383, 0.231621, 0.276939, 0.334340, 0.258812, 0.278953]
demographic_parity_latin_american_EO = [0.250000, 0.216667, 0.170833, 0.220833, 0.350000, 0.170833, 0.241667, 0.300000, 0.233333, 0.212500]
demographic_parity_arab_EO = [0.300000, 0.338889, 0.227778, 0.233333, 0.377778, 0.266667, 0.266667, 0.416667, 0.194444, 0.294444]
demographic_parity_unknown_EO = [0.242857, 0.302041, 0.228571, 0.224490, 0.334694, 0.204082, 0.295918, 0.265306, 0.265306, 0.273469]

# DP - Demographic Parity
demographic_parity_african_american_DP = [0.307263, 0.335196, 0.303538, 0.318436, 0.353818, 0.271881, 0.290503, 0.234637, 0.275605, 0.286778]
demographic_parity_asian_DP = [0.274322, 0.255183, 0.256778, 0.244019, 0.277512, 0.258373, 0.248804, 0.205742, 0.285486, 0.240829]
demographic_parity_caucasian_DP = [0.308157, 0.276939, 0.209466, 0.256798, 0.267875, 0.209466, 0.243706, 0.216516, 0.289023, 0.259819]
demographic_parity_latin_american_DP = [0.250000, 0.250000, 0.233333, 0.220833, 0.266667, 0.195833, 0.212500, 0.266667, 0.237500, 0.212500]
demographic_parity_arab_DP = [0.261111, 0.338889, 0.383333, 0.355556, 0.322222, 0.277778, 0.288889, 0.338889, 0.233333, 0.261111]
demographic_parity_unknown_DP = [0.271429, 0.310204, 0.261224, 0.275510, 0.302041, 0.244898, 0.265306, 0.232653, 0.295918, 0.251020]

# Equal Opportunity
equal_opportunity_african_american_EO = [0.682927, 0.829268, 0.621951, 0.731707, 0.780488, 0.707317, 0.707317, 0.695122, 0.634146, 0.743902]
equal_opportunity_asian_EO = [0.737226, 0.751825, 0.598540, 0.708029, 0.773723, 0.664234, 0.810219, 0.729927, 0.635036, 0.766423]
equal_opportunity_caucasian_EO = [0.722222, 0.772222, 0.572222, 0.777778, 0.805556, 0.655556, 0.777778, 0.750000, 0.700000, 0.777778]
equal_opportunity_latin_american_EO = [0.729167, 0.645833, 0.458333, 0.604167, 0.625000, 0.541667, 0.708333, 0.750000, 0.770833, 0.645833]
equal_opportunity_arab_EO = [0.607143, 0.714286, 0.464286, 0.500000, 0.750000, 0.607143, 0.678571, 0.642857, 0.392857, 0.642857]
equal_opportunity_unknown_EO = [0.808219, 0.739726, 0.726027, 0.630137, 0.726027, 0.643836, 0.821918, 0.739726, 0.657534, 0.780822]

# Predictive Parity
predictive_parity_african_american_EO = [0.400000, 0.365591, 0.459459, 0.451128, 0.318408, 0.417266, 0.353659, 0.390411, 0.393939, 0.393548]
predictive_parity_asian_EO = [0.615854, 0.530928, 0.611940, 0.642384, 0.566845, 0.590909, 0.502262, 0.515464, 0.564935, 0.632530]
predictive_parity_caucasian_EO = [0.465950, 0.477663, 0.575419, 0.544747, 0.428994, 0.513043, 0.509091, 0.406627, 0.490272, 0.505415]
predictive_parity_latin_american_EO = [0.583333, 0.596154, 0.536585, 0.547170, 0.357143, 0.634146, 0.586207, 0.500000, 0.660714, 0.607843]
predictive_parity_arab_EO = [0.314815, 0.327869, 0.317073, 0.333333, 0.308824, 0.354167, 0.395833, 0.240000, 0.314286, 0.339623]
predictive_parity_unknown_EO = [0.495798, 0.364865, 0.473214, 0.418182, 0.323171, 0.470000, 0.379310, 0.415385, 0.369231, 0.425373]

# Equal Opportunity (DP only)
equal_opportunity_african_american_DP = [0.829268, 0.841463, 0.804878, 0.829268, 0.853659, 0.768293, 0.756098, 0.682927, 0.756098, 0.768293]
equal_opportunity_asian_DP = [0.759124, 0.759124, 0.744526, 0.729927, 0.729927, 0.766423, 0.759124, 0.656934, 0.788321, 0.715328]
equal_opportunity_caucasian_DP = [0.833333, 0.811111, 0.688889, 0.772222, 0.805556, 0.655556, 0.744444, 0.700000, 0.822222, 0.766667]
equal_opportunity_latin_american_DP = [0.729167, 0.770833, 0.729167, 0.687500, 0.812500, 0.604167, 0.645833, 0.791667, 0.770833, 0.645833]
equal_opportunity_arab_DP = [0.642857, 0.750000, 0.821429, 0.857143, 0.714286, 0.714286, 0.678571, 0.750000, 0.607143, 0.642857]
equal_opportunity_unknown_DP = [0.835616, 0.835616, 0.821918, 0.835616, 0.835616, 0.794521, 0.821918, 0.767123, 0.849315, 0.835616]

# Predictive Parity (DP only)
predictive_parity_african_american_DP = [0.412121, 0.383333, 0.404908, 0.397661, 0.368421, 0.431507, 0.397436, 0.444444, 0.418919, 0.409091]
predictive_parity_asian_DP = [0.604651, 0.650000, 0.633540, 0.653595, 0.574713, 0.648148, 0.666667, 0.697674, 0.603352, 0.649007]
predictive_parity_caucasian_DP = [0.490196, 0.530909, 0.596154, 0.545098, 0.545113, 0.567308, 0.553719, 0.586047, 0.515679, 0.534884]
predictive_parity_latin_american_DP = [0.583333, 0.616667, 0.625000, 0.622642, 0.609375, 0.617021, 0.607843, 0.593750, 0.649123, 0.634615]
predictive_parity_arab_DP = [0.382979, 0.344262, 0.333333, 0.375000, 0.344828, 0.400000, 0.365385, 0.344262, 0.404762, 0.404255]
predictive_parity_unknown_DP = [0.458647, 0.401316, 0.468750, 0.451852, 0.412162, 0.483333, 0.461538, 0.491228, 0.427586, 0.495935]





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
