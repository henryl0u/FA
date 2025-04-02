import numpy as np
import sys
from scipy.stats import ttest_rel

base_path = "./Fairness/RF/result"

# Save the default standard output
default_stdout = sys.stdout

# Open the file you want to redirect output to
f = open(f"{base_path}/prints.txt", "w")

# Change standard output to point to the file
sys.stdout = f

accuracy = [
    0.8144766873165961,
    0.8278447994783176,
    0.818063253994131,
    0.8301271600912944,
    0.817411150961852,
    0.8275187479621781,
    0.820345614607108,
    0.8373002934463645,
    0.823932181284643,
    0.8415389631561787
]

balanced_accuracy = [
    0.8128109578879349,
    0.8209491079474823,
    0.8149943640014721,
    0.8116297887876953,
    0.8131695464832238,
    0.8143253608343016,
    0.8163838042555411,
    0.8152826837205125,
    0.8171393757805641,
    0.8050125614671562
]

f1_score = [
    0.609471516815374,
    0.6271186440677966,
    0.6141078838174274,
    0.6221899927483684,
    0.6121883656509696,
    0.6218727662616155,
    0.6170952050034746,
    0.6317343173431734,
    0.6207865168539326,
    0.6278713629402757
]

roc_auc_score = [
    0.8871242788384917,
    0.887356818109376,
    0.8877835023166996,
    0.8874988047046825,
    0.889089634109237,
    0.8884434502163123,
    0.8872474304772777,
    0.8916555347244156,
    0.8868120532130986,
    0.8877668406243933
]


demographic_parity_african_american = [0.318436, 0.292365, 0.297952, 0.281192, 0.301676, 0.283054, 0.296089, 0.271881, 0.297952, 0.253259]
demographic_parity_asian = [0.322169, 0.301435, 0.314195, 0.293461, 0.323764, 0.314195, 0.317384, 0.295056, 0.314195, 0.267943]
demographic_parity_caucasian = [0.303122, 0.291037, 0.301108, 0.279960, 0.305136, 0.281974, 0.299094, 0.271903, 0.294058, 0.263847]
demographic_parity_latin_american = [0.275000, 0.287500, 0.266667, 0.245833, 0.270833, 0.275000, 0.295833, 0.254167, 0.283333, 0.233333]
demographic_parity_arab = [0.266667, 0.261111, 0.250000, 0.250000, 0.272222, 0.233333, 0.244444, 0.200000, 0.250000, 0.222222]
demographic_parity_unknown = [0.246939, 0.238776, 0.240816, 0.232653, 0.232653, 0.232653, 0.246939, 0.222449, 0.232653, 0.195918]

equal_opportunity_african_american = [0.853659, 0.792683, 0.804878, 0.780488, 0.817073, 0.780488, 0.792683, 0.780488, 0.817073, 0.731707]
equal_opportunity_asian = [0.824818, 0.824818, 0.810219, 0.802920, 0.817518, 0.824818, 0.832117, 0.810219, 0.810219, 0.773723]
equal_opportunity_caucasian = [0.816667, 0.811111, 0.827778, 0.805556, 0.827778, 0.805556, 0.822222, 0.794444, 0.827778, 0.783333]
equal_opportunity_latin_american = [0.812500, 0.875000, 0.791667, 0.770833, 0.812500, 0.812500, 0.854167, 0.791667, 0.854167, 0.729167]
equal_opportunity_arab = [0.642857, 0.678571, 0.642857, 0.607143, 0.678571, 0.642857, 0.642857, 0.607143, 0.678571, 0.607143]
equal_opportunity_unknown = [0.780822, 0.808219, 0.753425, 0.767123, 0.767123, 0.767123, 0.794521, 0.753425, 0.753425, 0.698630]

predictive_parity_african_american = [0.409357, 0.414013, 0.412500, 0.423841, 0.413580, 0.421053, 0.408805, 0.438356, 0.418750, 0.441176]
predictive_parity_asian = [0.559406, 0.597884, 0.563452, 0.597826, 0.551724, 0.573604, 0.572864, 0.600000, 0.563452, 0.630952]
predictive_parity_caucasian = [0.488372, 0.505190, 0.498328, 0.521583, 0.491749, 0.517857, 0.498316, 0.529630, 0.510274, 0.538168]
predictive_parity_latin_american = [0.590909, 0.608696, 0.593750, 0.627119, 0.600000, 0.590909, 0.577465, 0.622951, 0.602941, 0.625000]
predictive_parity_arab = [0.375000, 0.404255, 0.400000, 0.377778, 0.387755, 0.428571, 0.409091, 0.472222, 0.422222, 0.425000]
predictive_parity_unknown = [0.471074, 0.504274, 0.466102, 0.491228, 0.491228, 0.491228, 0.479339, 0.504587, 0.482456, 0.531250]



def print_perf_stats(metric_name, values):
    print(f"{metric_name} - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")

print_perf_stats("Accuracy", accuracy)
print_perf_stats("Balanced Accuracy", balanced_accuracy)
print_perf_stats("F1 Score", f1_score)
print_perf_stats("ROC AUC Score", roc_auc_score)


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

# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()