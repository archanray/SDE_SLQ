import numpy as np
import pickle
import os

def sort_abs_descending(x, type="values"):
    """
    Sort array by absolute value in descending order.
    """
    abs_x = np.abs(x)
    idx = np.argsort(-abs_x)
    if type == "values":
        return x[idx]
    else:
        return idx

def sort_descending(x, type="values"):
    """
    Sort array in descending order.
    """
    idx = np.argsort(-x)
    if type == "values":
        return x[idx]
    else:
        return idx

def get_spectrum(input):
    """
    get spectrum of the input matrix
    """
    spectrum, _ = np.linalg.eig(input)
    spectrum = sort_abs_descending(np.real(spectrum))
    return spectrum

def saver(save_dict, name_adder="single_eval_method"):
    dir_path = os.path.join("outputs", save_dict["dataset"])
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    if "single_eval_method_" in name_adder:
        file_path = os.path.join(dir_path, \
                                name_adder+
                                save_dict["method"]+\
                                ".pkl")
    with open(file_path, "wb") as f:
        pickle.dump(save_dict, f)
    return None