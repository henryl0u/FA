import numpy as np
import sys
from scipy.stats import ttest_rel
import json

base_path = "./Fairness/RF/reweight/result"

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
    0.8415389631561787,
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
    0.8050125614671562,
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
    0.6278713629402757,
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
    0.8877668406243933,
]


demographic_parity_african_american = [
    0.318436,
    0.292365,
    0.297952,
    0.281192,
    0.301676,
    0.283054,
    0.296089,
    0.271881,
    0.297952,
    0.253259,
]
demographic_parity_asian = [
    0.322169,
    0.301435,
    0.314195,
    0.293461,
    0.323764,
    0.314195,
    0.317384,
    0.295056,
    0.314195,
    0.267943,
]
demographic_parity_caucasian = [
    0.303122,
    0.291037,
    0.301108,
    0.279960,
    0.305136,
    0.281974,
    0.299094,
    0.271903,
    0.294058,
    0.263847,
]
demographic_parity_latin_american = [
    0.275000,
    0.287500,
    0.266667,
    0.245833,
    0.270833,
    0.275000,
    0.295833,
    0.254167,
    0.283333,
    0.233333,
]
demographic_parity_arab = [
    0.266667,
    0.261111,
    0.250000,
    0.250000,
    0.272222,
    0.233333,
    0.244444,
    0.200000,
    0.250000,
    0.222222,
]
demographic_parity_unknown = [
    0.246939,
    0.238776,
    0.240816,
    0.232653,
    0.232653,
    0.232653,
    0.246939,
    0.222449,
    0.232653,
    0.195918,
]

equal_opportunity_african_american = [
    0.853659,
    0.792683,
    0.804878,
    0.780488,
    0.817073,
    0.780488,
    0.792683,
    0.780488,
    0.817073,
    0.731707,
]
equal_opportunity_asian = [
    0.824818,
    0.824818,
    0.810219,
    0.802920,
    0.817518,
    0.824818,
    0.832117,
    0.810219,
    0.810219,
    0.773723,
]
equal_opportunity_caucasian = [
    0.816667,
    0.811111,
    0.827778,
    0.805556,
    0.827778,
    0.805556,
    0.822222,
    0.794444,
    0.827778,
    0.783333,
]
equal_opportunity_latin_american = [
    0.812500,
    0.875000,
    0.791667,
    0.770833,
    0.812500,
    0.812500,
    0.854167,
    0.791667,
    0.854167,
    0.729167,
]
equal_opportunity_arab = [
    0.642857,
    0.678571,
    0.642857,
    0.607143,
    0.678571,
    0.642857,
    0.642857,
    0.607143,
    0.678571,
    0.607143,
]
equal_opportunity_unknown = [
    0.780822,
    0.808219,
    0.753425,
    0.767123,
    0.767123,
    0.767123,
    0.794521,
    0.753425,
    0.753425,
    0.698630,
]

predictive_parity_african_american = [
    0.409357,
    0.414013,
    0.412500,
    0.423841,
    0.413580,
    0.421053,
    0.408805,
    0.438356,
    0.418750,
    0.441176,
]
predictive_parity_asian = [
    0.559406,
    0.597884,
    0.563452,
    0.597826,
    0.551724,
    0.573604,
    0.572864,
    0.600000,
    0.563452,
    0.630952,
]
predictive_parity_caucasian = [
    0.488372,
    0.505190,
    0.498328,
    0.521583,
    0.491749,
    0.517857,
    0.498316,
    0.529630,
    0.510274,
    0.538168,
]
predictive_parity_latin_american = [
    0.590909,
    0.608696,
    0.593750,
    0.627119,
    0.600000,
    0.590909,
    0.577465,
    0.622951,
    0.602941,
    0.625000,
]
predictive_parity_arab = [
    0.375000,
    0.404255,
    0.400000,
    0.377778,
    0.387755,
    0.428571,
    0.409091,
    0.472222,
    0.422222,
    0.425000,
]
predictive_parity_unknown = [
    0.471074,
    0.504274,
    0.466102,
    0.491228,
    0.491228,
    0.491228,
    0.479339,
    0.504587,
    0.482456,
    0.531250,
]

# Reweighted Model Metrics (New)

accuracy_reweighted = [
    0.8187153570264102,
    0.8278447994783176,
    0.8301271600912944,
    0.8337137267688295,
    0.817411150961852,
    0.8343658298011086,
    0.8447994783175742,
    0.836974241930225,
    0.8294750570590154,
    0.8131724812520378,
]

balanced_accuracy_reweighted = [
    0.8196748506967486,
    0.8188073560647111,
    0.8159132925532377,
    0.8138131949012324,
    0.8117417118947097,
    0.8142101778309665,
    0.8119948971756259,
    0.8207955306097021,
    0.8190858960947891,
    0.822725751442323,
]

f1_score_reweighted = [
    0.6181318681318682,
    0.625531914893617,
    0.6254493170381021,
    0.6271929824561403,
    0.6111111111111112,
    0.6281112737920937,
    0.6366412213740458,
    0.6355685131195336,
    0.6272273699215966,
    0.6156941649899397,
]

roc_auc_score_reweighted = [
    0.8876400668785841,
    0.8863419037214976,
    0.8875821131662142,
    0.887534301353509,
    0.8865882069990698,
    0.8872807538618904,
    0.8883825988183237,
    0.8904863185773522,
    0.8869033303100814,
    0.8905993283164736,
]

# Demographic Parity (Reweighted)
demographic_parity_african_american_reweighted = [
    0.318436,
    0.294227,
    0.277467,
    0.279330,
    0.294227,
    0.266294,
    0.245810,
    0.268156,
    0.288641,
    0.327747,
]
demographic_parity_asian_reweighted = [
    0.312600,
    0.307815,
    0.293461,
    0.288676,
    0.325359,
    0.301435,
    0.275917,
    0.291866,
    0.312600,
    0.325359,
]
demographic_parity_caucasian_reweighted = [
    0.298087,
    0.284995,
    0.287009,
    0.272910,
    0.301108,
    0.266868,
    0.256798,
    0.278953,
    0.275932,
    0.321249,
]
demographic_parity_latin_american_reweighted = [
    0.275000,
    0.279167,
    0.266667,
    0.250000,
    0.279167,
    0.266667,
    0.229167,
    0.266667,
    0.283333,
    0.295833,
]
demographic_parity_arab_reweighted = [
    0.283333,
    0.255556,
    0.266667,
    0.261111,
    0.283333,
    0.250000,
    0.233333,
    0.238889,
    0.261111,
    0.266667,
]
demographic_parity_unknown_reweighted = [
    0.261224,
    0.234694,
    0.230612,
    0.226531,
    0.230612,
    0.228571,
    0.214286,
    0.230612,
    0.234694,
    0.255102,
]

# Equal Opportunity (Reweighted)
equal_opportunity_african_american_reweighted = [
    0.853659,
    0.817073,
    0.780488,
    0.780488,
    0.792683,
    0.768293,
    0.731707,
    0.780488,
    0.817073,
    0.865854,
]
equal_opportunity_asian_reweighted = [
    0.832117,
    0.832117,
    0.817518,
    0.802920,
    0.817518,
    0.810219,
    0.795620,
    0.810219,
    0.824818,
    0.846715,
]
equal_opportunity_caucasian_reweighted = [
    0.816667,
    0.805556,
    0.805556,
    0.805556,
    0.816667,
    0.794444,
    0.783333,
    0.800000,
    0.805556,
    0.855556,
]
equal_opportunity_latin_american_reweighted = [
    0.833333,
    0.812500,
    0.791667,
    0.770833,
    0.833333,
    0.791667,
    0.729167,
    0.812500,
    0.833333,
    0.854167,
]
equal_opportunity_arab_reweighted = [
    0.678571,
    0.678571,
    0.714286,
    0.642857,
    0.714286,
    0.642857,
    0.642857,
    0.678571,
    0.678571,
    0.678571,
]
equal_opportunity_unknown_reweighted = [
    0.821918,
    0.780822,
    0.767123,
    0.753425,
    0.767123,
    0.767123,
    0.739726,
    0.808219,
    0.767123,
    0.794521,
]

# Predictive Parity (Reweighted)
predictive_parity_african_american_reweighted = [
    0.409357,
    0.424051,
    0.429530,
    0.426667,
    0.411392,
    0.440559,
    0.454545,
    0.444444,
    0.432258,
    0.403409,
]
predictive_parity_asian_reweighted = [
    0.581633,
    0.590674,
    0.608696,
    0.607735,
    0.549020,
    0.587302,
    0.630058,
    0.606557,
    0.576531,
    0.568627,
]
predictive_parity_caucasian_reweighted = [
    0.496622,
    0.512367,
    0.508772,
    0.535055,
    0.491639,
    0.539623,
    0.552941,
    0.519856,
    0.529197,
    0.482759,
]
predictive_parity_latin_american_reweighted = [
    0.606061,
    0.582090,
    0.593750,
    0.616667,
    0.597015,
    0.593750,
    0.636364,
    0.609375,
    0.588235,
    0.577465,
]
predictive_parity_arab_reweighted = [
    0.372549,
    0.413043,
    0.416667,
    0.382979,
    0.392157,
    0.400000,
    0.428571,
    0.441860,
    0.404255,
    0.395833,
]
predictive_parity_unknown_reweighted = [
    0.468750,
    0.495652,
    0.495575,
    0.495495,
    0.495575,
    0.500000,
    0.514286,
    0.522124,
    0.486957,
    0.464000,
]


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
    print(
        f"{metric_name} - ΔMean: {mean_diff:.4f}, ΔStd: {std_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}"
    )


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

# Now define reweighted fairness metrics in the same structure
fairness_metrics_reweighted = {
    "Demographic Parity": {
        "African American": demographic_parity_african_american_reweighted,
        "Asian": demographic_parity_asian_reweighted,
        "Caucasian": demographic_parity_caucasian_reweighted,
        "Latin American": demographic_parity_latin_american_reweighted,
        "Arab": demographic_parity_arab_reweighted,
        "Unknown/Other": demographic_parity_unknown_reweighted,
    },
    "Equal Opportunity": {
        "African American": equal_opportunity_african_american_reweighted,
        "Asian": equal_opportunity_asian_reweighted,
        "Caucasian": equal_opportunity_caucasian_reweighted,
        "Latin American": equal_opportunity_latin_american_reweighted,
        "Arab": equal_opportunity_arab_reweighted,
        "Unknown/Other": equal_opportunity_unknown_reweighted,
    },
    "Predictive Parity": {
        "African American": predictive_parity_african_american_reweighted,
        "Asian": predictive_parity_asian_reweighted,
        "Caucasian": predictive_parity_caucasian_reweighted,
        "Latin American": predictive_parity_latin_american_reweighted,
        "Arab": predictive_parity_arab_reweighted,
        "Unknown/Other": predictive_parity_unknown_reweighted,
    },
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
        print(
            f"{group} - ΔMean: {mean_diff:.4f}, ΔStd: {std_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}"
        )

    print(f"\n--- Difference vs Baseline Values: {metric_name} ---")
    baseline_groups = fairness_metrics[metric_name]
    for group, reweighted_values in groups.items():
        baseline_values = np.array(baseline_groups[group])
        reweighted_values = np.array(reweighted_values)
        diff = reweighted_values - baseline_values
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        t_stat, p_val = ttest_rel(reweighted_values, baseline_values)
        print(
            f"{group} - ΔMean: {mean_diff:.4f}, ΔStd: {std_diff:.4f}, t={t_stat:.3f}, p={p_val:.4f}"
        )

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
with open("./Fairness/RF/base/result/fairness_differences.json") as f:
    baseline_diffs = json.load(f)

with open("./Fairness/RF/reweight/result/fairness_differences.json") as f:
    mitigated_diffs = json.load(f)

# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== Disparity Comparison: {metric_name} ===")

    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs[metric_name][group])

        t_stat, p_val = ttest_rel(mitigated_diff, base_diff)

        print(f"{group} - t={t_stat:.3f}, p={p_val:.4f}")

# Iterate through all metrics and groups
for metric_name in baseline_diffs.keys():
    print(f"\n=== Disparity Comparison (ABS): {metric_name} ===")

    for group in baseline_diffs[metric_name]:
        base_diff = np.array(baseline_diffs[metric_name][group])
        mitigated_diff = np.array(mitigated_diffs[metric_name][group])

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

        print(
            f"{group} - ΔΔMean: {mean_delta:.4f}, ΔΔStd: {std_delta:.4f}, t={t_stat:.3f}, p={p_val:.4f} → {interpretation}"
        )

# === Aggregated Disparities vs Reference Group (Reweighted) ===
print("\n=== Aggregated Fairness Disparities (vs Caucasian, Reweighted) ===")

for metric_name, groups in fairness_metrics_reweighted.items():
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
        f"{metric_name} Disparity (vs {reference_group}): {mean_disp:.4f} ± {std_disp:.4f}"
    )


# Change standard output back to default
sys.stdout = default_stdout

# Close the file
f.close()
