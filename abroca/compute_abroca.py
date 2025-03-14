from abroca.utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn import preprocessing


def compute_abroca(
    df,
    pred_col,
    label_col,
    protected_attr_col,
    majority_protected_attr_val,
    n_grid=10000,
    plot_slices=True,
    lb=0, 
    ub=1,
    limit=1000,
    file_path=None,
):
    """
    Compute the value of the ABROCA statistic for each minority group
    compared to the specified majority protected attribute group,
    as well as the majority group against itself.
    """
    # Validations
    if not df[pred_col].between(0, 1, inclusive='both').all():
        print("Predictions must be in range [0,1].")
        exit(1)
    if len(df[label_col].value_counts()) != 2:
        print("The label column should be binary.")
        exit(1)
    if len(df[protected_attr_col].value_counts()) < 2:
        print("The protected attribute column must have at least two distinct values.")
        exit(1)

    # Initialize variables
    prot_attr_values = df[protected_attr_col].value_counts().index.values
    fpr_tpr_dict = {}
    minority_protected_attr_vals = [val for val in prot_attr_values if val != majority_protected_attr_val]

    # Compute ROC for each protected attribute value
    for pa_value in prot_attr_values:
        pa_df = df[df[protected_attr_col] == pa_value]
        fpr_tpr_dict[pa_value] = compute_roc(pa_df[pred_col], pa_df[label_col])

    # Check if minority protected attr values are found
    if not minority_protected_attr_vals:
        print("No minority protected attribute values found.")
        exit(1)

    results = {}

    # Compare the majority group with itself
    majority_roc_x, majority_roc_y = interpolate_roc_fun(
        fpr_tpr_dict[majority_protected_attr_val][0],
        fpr_tpr_dict[majority_protected_attr_val][1],
        n_grid,
    )
    # Computing ABROCA for the majority group against itself
    f1_self = interpolate.interp1d(x=majority_roc_x, y=(majority_roc_y - majority_roc_y))  # This will always be 0
    f2_self = lambda x, _: abs(f1_self(x))
    slice_statistic_self, _ = integrate.quad(f2_self, lb, ub, limit)
    results[majority_protected_attr_val] = slice_statistic_self

    # Compare each minority class to the majority class
    for minority_protected_attr_val in minority_protected_attr_vals:
        minority_roc_x, minority_roc_y = interpolate_roc_fun(
            fpr_tpr_dict[minority_protected_attr_val][0],
            fpr_tpr_dict[minority_protected_attr_val][1],
            n_grid,
        )

        # Check if ROC curves share the same x-values
        if list(majority_roc_x) == list(minority_roc_x):
            f1 = interpolate.interp1d(x=majority_roc_x, y=(majority_roc_y - minority_roc_y))
            f2 = lambda x, _: abs(f1(x))
            slice_statistic, _ = integrate.quad(f2, lb, ub, limit)
            results[minority_protected_attr_val] = slice_statistic
            fig_name = str(minority_protected_attr_val).replace("/", "_").replace(" ", "_")
            
            if plot_slices:
                slice_plot(
                    majority_roc_x,
                    minority_roc_x,
                    majority_roc_y,
                    minority_roc_y,
                    majority_group_name=majority_protected_attr_val,
                    minority_group_name=minority_protected_attr_val,
                    fout=f"{file_path}/slice_{fig_name}.png",
                )
        else:
            print(f"Majority and minority FPR are different for {minority_protected_attr_val}.")

    return results