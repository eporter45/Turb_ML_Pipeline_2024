import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import pickle

def load_dict_from_file(file_name):
    with open(file_name, 'rb') as file:
        loaded_dict = pickle.load(file)
    print(f"Dictionary loaded from {file_name}")
    return loaded_dict



def slice_y_tensors(y_pred_list, y_true_list, test_cases):
    slice_indices_dict = load_dict_from_file("slice_indices_dict.pkl")
    sliced_pred = []
    sliced_true = []
    for case, y_pred, y_true in zip(test_cases, y_pred_list, y_true_list):
        if 'duct' in case.lower():
            sliced_pred.append(y_pred)
            sliced_true.append(y_true)
        else:
            slice_indices = slice_indices_dict[case]
            y_pred_slice = y_pred[slice_indices]
            y_true_slice = y_true[slice_indices]
            sliced_pred.append(y_pred_slice)
            sliced_true.append(y_true_slice)
    return sliced_pred, sliced_true



'''make tke from trace of reynolds stress tensor (xx + yy + zz) * 1/2'''
def check_shapes(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError('tensor1 and tensor2 must have same shape')
    else:
        return True
def make_norm_factor_tke(y_pred_tensor, y_true_tensor):
    # the 0th, 3rd and 5th index of the y_tensors are the xx, yy, and zz components of the Reynolds Stress Tensor
    trace_pred = y_pred_tensor[:, 0] + y_pred_tensor[:, 3] + y_pred_tensor[:, 5]
    trace_true = y_true_tensor[:, 0] + y_true_tensor[:, 3] + y_true_tensor[:, 5]
    tke_pred = 0.5 * trace_pred
    tke_true = 0.5 * trace_true
    print(tke_pred, tke_true)
    # print ratio
    return tke_pred, tke_true


def reorder_to_matrix(tensor):
    """
    Reorders a tensor of shape (n, 6) where the columns correspond to [xx, xy, xz, yy, yz, zz]
    into a full (n, 3, 3) symmetric matrix.
    """
    n = tensor.shape[0]
    full_matrix = torch.zeros((n, 3, 3))

    # Fill the matrix using the input format
    full_matrix[:, 0, 0] = tensor[:, 0]  # xx
    full_matrix[:, 0, 1] = tensor[:, 1]  # xy
    full_matrix[:, 0, 2] = tensor[:, 2]  # xz
    full_matrix[:, 1, 1] = tensor[:, 3]  # yy
    full_matrix[:, 1, 2] = tensor[:, 4]  # yz
    full_matrix[:, 2, 2] = tensor[:, 5]  # zz

    # Since it's symmetric, we can copy the off-diagonal elements
    full_matrix[:, 1, 0] = full_matrix[:, 0, 1]  # xy
    full_matrix[:, 2, 0] = full_matrix[:, 0, 2]  # xz
    full_matrix[:, 2, 1] = full_matrix[:, 1, 2]  # yz

    return full_matrix


def compute_frobenius_norm(y_true, y_pred):
    """
    Reorders tensors, computes the absolute difference, and then calculates the Frobenius norm.
    """
    # Reorder y_true and y_pred into (n, 3, 3) matrices
    y_true_matrix = reorder_to_matrix(y_true)
    y_pred_matrix = reorder_to_matrix(y_pred)

    # Compute the absolute difference
    diff = torch.abs(y_true_matrix - y_pred_matrix)

    # Compute Frobenius norm for each matrix in the batch (n, 3, 3)
    frobenius_norms = torch.norm(diff, p='fro', dim=(1, 2))

    return frobenius_norms


def compute_l1_norm(y_true, y_pred):
    """
    Reorders tensors, computes the absolute difference, and then calculates the L1 norm.
    """
    # Reorder y_true and y_pred into (n, 3, 3) matrices
    y_true_matrix = reorder_to_matrix(y_true)
    y_pred_matrix = reorder_to_matrix(y_pred)

    # Compute the absolute difference
    diff = torch.abs(y_true_matrix - y_pred_matrix)

    # Compute L1 norm by summing all elements
    l1_norms = torch.sum(diff, dim=(1, 2))  # Sum over the matrix dimensions (1, 2)

    return l1_norms
def make_abs_diff_metrics(y_pred_tensor, y_true_tensor):
    ''' returns metrics that use abs diff'''

    assert y_pred_tensor.shape == y_true_tensor.shape
    num_features = y_true_tensor.shape[1]
    num_points = y_true_tensor.shape[0]
    #do tke stuff
    # sum sq diffs emphasizes wrong errors more if (diff > 1 or diff < 1
    tke_pred, tke_true = make_norm_factor_tke(y_pred_tensor, y_true_tensor)
    abs_diff_tke = torch.abs(tke_pred - tke_true)
    sum_abs_diff_tke = torch.sum(abs_diff_tke, dim=0)
    sum_abs_diff_sq_tke = torch.sum(torch.square(abs_diff_tke), dim=0)
    L2_tke = torch.sqrt(sum_abs_diff_sq_tke)
    var_abs_diff_tke = sum_abs_diff_sq_tke / num_points
    ratio_tke = torch.sum(torch.abs(tke_pred), dim=0) / torch.sum(torch.abs(tke_true), dim=0)
    mean_abs_diff_ratio_tke = sum_abs_diff_tke / num_points
    median_abs_diff_tke = torch.median(sum_abs_diff_tke)
    # now do regular absolute difference
    # make sum of abs diff for each column
    #remember the metrics below are for the 6 components of Rey stress, not all 9
    abs_diff = torch.abs(y_pred_tensor - y_true_tensor)
    sum_abs_diff = torch.sum(abs_diff, dim=0)

    abs_diff_sq = torch.square(abs_diff)
    sum_abs_diff_sq = torch.sum(abs_diff_sq, dim=0)
    L2_rey = torch.sqrt(sum_abs_diff_sq)
    var_abs_diff_rey = sum_abs_diff_sq / num_points

    ratio_rey = torch.sum(torch.abs(y_pred_tensor), dim=0) / torch.sum(torch.abs(y_true_tensor), dim=0)
    # make mean per component
    mean_per_point = sum_abs_diff / num_points
    # make global sum of means per point
    sum_per_component = torch.sum(mean_per_point, dim=0)
    # average global sum of means per point per component
    avg_per_component = sum_per_component / num_features
    #median of abs diff
    median_of_points= torch.median(torch.abs(y_pred_tensor - y_true_tensor), dim=0)
    frobenius_norms = compute_frobenius_norm(y_pred_tensor, y_true_tensor)
    avg_frobenius_norms = torch.sum(frobenius_norms)/ num_points
    median_frobenius_norms = torch.median(frobenius_norms)
    max_frobenius_norms = torch.max(frobenius_norms)
    min_frobenius_norms = torch.min(frobenius_norms)
    L1_norms = compute_l1_norm(y_pred_tensor, y_true_tensor)

    avg_l1_norms = torch.sum(L1_norms)/ num_points
    print(avg_l1_norms)
    median_l1_norms = torch.median(L1_norms)
    max_l1_norms = torch.max(L1_norms)
    min_l1_norms = torch.min(L1_norms)
    # divide n points -> abs error per point per component
    # ^ this is mean per point
    #  sum this up divide by 6, thats average per component
    # make the matrix, non upper triagular (full 3x3)
    # for doing median: do abs difference, take median of each column
    # for
    returned_features = [ratio_tke, sum_abs_diff_tke, sum_abs_diff_sq_tke, L2_tke, var_abs_diff_tke, mean_abs_diff_ratio_tke, median_abs_diff_tke,
                         ratio_rey, sum_abs_diff, sum_abs_diff_sq, L2_rey, var_abs_diff_rey, mean_per_point, avg_per_component, median_of_points,
                         frobenius_norms, avg_frobenius_norms, median_frobenius_norms, max_frobenius_norms, min_frobenius_norms,
                         L1_norms, avg_l1_norms, median_l1_norms, min_l1_norms, max_l1_norms]


    return returned_features


def write_individual_metrics(f, metrics_list, case):
    """Writes the individual metrics to the file with proper formatting."""
    metric_names = [
        "Ratio of Pred / True TKE",
        "Sum of Abs Diff of TKE",
        "Sum of ABS Diff Squared TKE",
        "L2, sqrt(sum(abs diff sq)) TKE",
        "Variance of abs diff TKE",
        "Mean of Abs Diff of TKE (abs_diff/n_points)",
        "Median of Abs Diff of TKE",
        "Ratio of Pred / True of each Reynolds",
        "Sum of Abs Diff Reynolds",
        "Sum of square(abs_diff) Reynolds",
        "L2 of Reynolds",
        "Variance of Abs Diff Reynolds",
        "Mean of Abs Diff of Reynolds (abs_diff/n_points)",
        "Mean per component of mean of Abs Diff of Reynolds",
        "Median of Abs Diff of Reynolds",
        "Frobenius (L2) Norms",
        "Avg (L2) Frobenius norms",
        "Median Frobenius norms",
        "Max Frobenius norms",
        "Min Frobenius norms",
        "L1 Norms",
        "Avg L1 norms",
        "Median L1 norms",
        "Min L1 norms",
        "Max L1 norms"
    ]

    # Write header
    f.write(f'Metrics for test case {case}\n')

    # Track sections for TKE and Reynolds stress metrics
    for i, (metric_name, metric_value) in enumerate(zip(metric_names, metrics_list)):
        # Handle section headers
        if i == 0:
            f.write("METRICS FOR TKE\n")
        elif i == 7:
            f.write("\nMETRICS FOR REYNOLDS STRESS\nVectors are structured as: [xx, xy, xz, yy, yz, zz]\n")

        # Handle tensor or list output formatting
        if isinstance(metric_value, torch.Tensor):
            if metric_value.numel() == 1:  # If it's a scalar
                metric_value = round(metric_value.item(), 4)
            else:
                metric_value = np.around(metric_value.cpu().numpy(), 4)

        # Write the metric name and its value
        f.write(f"{metric_name}: \n{metric_value}\n")

    f.write("\n")





# Example usage in your main function
def write_full_metrics(y_pred, y_true, test_cases, file_title='full_tensors', directory_path='../'):
    metrics_dict = {}
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has length {len(y_true)}, but y_pred has length {len(y_pred)}.")

    for i, (y_true_tensor, y_pred_tensor) in enumerate(zip(y_true, y_pred)):
        case_metrics = make_abs_diff_metrics(y_pred_tensor, y_true_tensor)
        metrics_dict[test_cases[i]] = case_metrics
        print(f'case {i}: {test_cases[i]} metrics: \n {case_metrics}')

    total_y_pred = torch.cat(y_pred, dim=0)
    total_y_true = torch.cat(y_true, dim=0)
    total_metrics = make_abs_diff_metrics(total_y_pred, total_y_true)
    metrics_dict['TOTAL'] = total_metrics

    # Write the metrics to a text file
    file_path = os.path.join(directory_path, f'metrics_{file_title}.txt')
    with open(file_path, 'w') as f:
        for key, value in metrics_dict.items():
            f.write(f"{key} metrics:\n")
            write_individual_metrics(f, value, key)  # Call the function to write individual metrics
            f.write("\n")


def save_results(y_pred, y_true, test_cases, save_path= '../' ):
    write_full_metrics(y_pred, y_true, test_cases, directory_path=save_path, file_title='Full_tensors')
    y_pred_sliced, y_true_sliced = slice_y_tensors(y_pred, y_true, test_cases)
    write_full_metrics(y_pred_sliced, y_true_sliced, test_cases, directory_path=save_path, file_title= 'Sliced_tensors')

import os





