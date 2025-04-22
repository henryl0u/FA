import numpy as np
import sys
from scipy.stats import ttest_rel

base_path = "./Fairness/CB/threshold/result"

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

# Accuracy
accuracy_EO = [
    0.8373002934463645,
    0.7345940658624063,
    0.8288229540267362,
    0.8434952722530160,
    0.7981741115096185,
    0.8213237691555266,
    0.7743723508314314,
    0.7828496902510597,
    0.7440495598304532,
    0.7958917508966417
]

accuracy_DP = [
    0.8099119660906423,
    0.8219758721878057,
    0.8164329964134334,
    0.8281708509944571,
    0.8203456146071080,
    0.8464297358982720,
    0.8356700358656668,
    0.8353439843495273,
    0.8307792631235735,
    0.8271926964460384
]

# Balanced Accuracy
balanced_accuracy_EO = [
    0.7510301272373755,
    0.7391934436965195,
    0.7658590333900314,
    0.7619406380124194,
    0.7564817605178744,
    0.7827112485258024,
    0.7569841467619812,
    0.7971268722671203,
    0.7656532977111181,
    0.7765098390915176
]

balanced_accuracy_DP = [
    0.8028929044372259,
    0.8145205924028478,
    0.7990096434977383,
    0.7983022459961229,
    0.8006776237818853,
    0.7944255048492770,
    0.8028675496880642,
    0.8112360657542821,
    0.8091711025404009,
    0.8105572828981493
]

# F1 Score
f1_score_EO = [
    0.5753191489361702,
    0.5012254901960784,
    0.5823389021479713,
    0.5918367346938775,
    0.5504720406681191,
    0.5910447761194030,
    0.5361930294906166,
    0.5741687979539642,
    0.5273931366646598,
    0.5664819944598338
]

f1_score_DP = [
    0.5982081323225362,
    0.6171107994389902,
    0.6004258339247693,
    0.6099185788304959,
    0.6050179211469534,
    0.6241021548284118,
    0.6204819277108434,
    0.6267553584626755,
    0.6208911614317020,
    0.6187050359712231
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

# DP - Demographic Parity
demographic_parity_african_american_DP = [0.327747, 0.316574, 0.314711, 0.322160, 0.333333, 0.242086, 0.255121, 0.225326, 0.245810, 0.303538]
demographic_parity_asian_DP = [0.261563, 0.250399, 0.280702, 0.240829, 0.271132, 0.245614, 0.259968, 0.250399, 0.259968, 0.256778]
demographic_parity_caucasian_DP = [0.325277, 0.275932, 0.233635, 0.251762, 0.239678, 0.207452, 0.229607, 0.253776, 0.286002, 0.274924]
demographic_parity_latin_american_DP = [0.250000, 0.237500, 0.225000, 0.195833, 0.241667, 0.187500, 0.266667, 0.283333, 0.220833, 0.229167]
demographic_parity_arab_DP = [0.283333, 0.394444, 0.450000, 0.300000, 0.355556, 0.277778, 0.294444, 0.433333, 0.227778, 0.277778]
demographic_parity_unknown_DP = [0.263265, 0.304082, 0.304082, 0.261224, 0.281633, 0.244898, 0.275510, 0.263265, 0.302041, 0.285714]

# DP - Equal Opportunity
equal_opportunity_african_american_DP = [0.817073, 0.853659, 0.792683, 0.804878, 0.829268, 0.743902, 0.743902, 0.731707, 0.768293, 0.804878]
equal_opportunity_asian_DP = [0.773723, 0.759124, 0.766423, 0.744526, 0.781022, 0.751825, 0.759124, 0.766423, 0.759124, 0.751825]
equal_opportunity_caucasian_DP = [0.811111, 0.794444, 0.716667, 0.750000, 0.716667, 0.683333, 0.722222, 0.744444, 0.811111, 0.800000]
equal_opportunity_latin_american_DP = [0.770833, 0.770833, 0.708333, 0.604167, 0.750000, 0.562500, 0.812500, 0.812500, 0.687500, 0.750000]
equal_opportunity_arab_DP = [0.678571, 0.857143, 0.928571, 0.750000, 0.750000, 0.678571, 0.714286, 0.964286, 0.607143, 0.714286]
equal_opportunity_unknown_DP = [0.808219, 0.849315, 0.876712, 0.808219, 0.835616, 0.794521, 0.794521, 0.808219, 0.849315, 0.835616]

# DP - Predictive Parity
predictive_parity_african_american_DP = [0.380682, 0.411765, 0.384615, 0.381503, 0.379888, 0.469231, 0.445255, 0.495868, 0.477273, 0.404908]
predictive_parity_asian_DP = [0.646341, 0.662420, 0.596591, 0.675497, 0.629412, 0.668831, 0.638037, 0.668790, 0.638037, 0.639752]
predictive_parity_caucasian_DP = [0.452012, 0.521898, 0.556034, 0.540000, 0.542017, 0.597087, 0.570175, 0.531746, 0.514085, 0.527473]
predictive_parity_latin_american_DP = [0.616667, 0.649123, 0.629630, 0.617021, 0.620690, 0.600000, 0.609375, 0.573529, 0.622642, 0.654545]
predictive_parity_arab_DP = [0.372549, 0.338028, 0.320988, 0.388889, 0.328125, 0.380000, 0.377358, 0.346154, 0.414634, 0.400000]
predictive_parity_unknown_DP = [0.457364, 0.416107, 0.429530, 0.460938, 0.442029, 0.483333, 0.429630, 0.457364, 0.418919, 0.435714]

# EO - Demographic Parity
demographic_parity_african_american_EO = [0.189944, 0.404097, 0.232775, 0.212291, 0.299814, 0.279330, 0.316574, 0.327747, 0.379888, 0.312849]
demographic_parity_asian_EO = [0.210526, 0.347687, 0.234450, 0.210526, 0.248804, 0.259968, 0.322169, 0.320574, 0.350877, 0.277512]
demographic_parity_caucasian_EO = [0.214502, 0.343404, 0.225579, 0.219537, 0.275932, 0.261833, 0.305136, 0.358510, 0.364552, 0.307150]
demographic_parity_latin_american_EO = [0.154167, 0.212500, 0.195833, 0.170833, 0.245833, 0.191667, 0.245833, 0.291667, 0.254167, 0.208333]
demographic_parity_arab_EO = [0.244444, 0.500000, 0.294444, 0.183333, 0.316667, 0.294444, 0.322222, 0.405556, 0.372222, 0.327778]
demographic_parity_unknown_EO = [0.202041, 0.340816, 0.230612, 0.183673, 0.248980, 0.244898, 0.310204, 0.285714, 0.406122, 0.285714]

# EO - Equal Opportunity
equal_opportunity_african_american_EO = [0.621951, 0.865854, 0.634146, 0.670732, 0.719512, 0.756098, 0.695122, 0.804878, 0.780488, 0.768293]
equal_opportunity_asian_EO = [0.620438, 0.693431, 0.642336, 0.613139, 0.737226, 0.715328, 0.744526, 0.824818, 0.802920, 0.708029]
equal_opportunity_caucasian_EO = [0.650000, 0.738889, 0.677778, 0.688889, 0.694444, 0.744444, 0.744444, 0.833333, 0.794444, 0.794444]
equal_opportunity_latin_american_EO = [0.458333, 0.666667, 0.541667, 0.541667, 0.666667, 0.625000, 0.770833, 0.812500, 0.750000, 0.687500]
equal_opportunity_arab_EO = [0.428571, 0.714286, 0.714286, 0.607143, 0.535714, 0.678571, 0.607143, 0.857143, 0.714286, 0.678571]
equal_opportunity_unknown_EO = [0.698630, 0.794521, 0.794521, 0.575342, 0.726027, 0.726027, 0.726027, 0.780822, 0.890411, 0.739726]

# EO - Predictive Parity
predictive_parity_african_american_EO = [0.500000, 0.327189, 0.416000, 0.482456, 0.366460, 0.413333, 0.335294, 0.375000, 0.313725, 0.375000]
predictive_parity_asian_EO = [0.643939, 0.435780, 0.598639, 0.636364, 0.647436, 0.601227, 0.504950, 0.562189, 0.500000, 0.557471]
predictive_parity_caucasian_EO = [0.549296, 0.390029, 0.544643, 0.568807, 0.456204, 0.515385, 0.442244, 0.421348, 0.395028, 0.468852]
predictive_parity_latin_american_EO = [0.594595, 0.627451, 0.553191, 0.634146, 0.542373, 0.652174, 0.627119, 0.557143, 0.590164, 0.660000]
predictive_parity_arab_EO = [0.272727, 0.222222, 0.377358, 0.515152, 0.263158, 0.358491, 0.293103, 0.328767, 0.298507, 0.322034]
predictive_parity_unknown_EO = [0.515152, 0.347305, 0.513274, 0.466667, 0.385246, 0.441667, 0.348684, 0.407143, 0.326633, 0.385714]



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



# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()