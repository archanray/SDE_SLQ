import numpy as np
import os
import argparse
from src.get_dataset import get_data
from src.utils import get_spectrum, saver
from src.approx_wrapper import SDE

def main(args):
    data, data_size = get_data(args.dataset)
    true_spectrum_sorted = get_spectrum(data)
    saveData = {"true_spectrum": true_spectrum_sorted, \
                "dataset": args.dataset, \
                "trials": args.trials,\
                "method": args.method}
    params = {"data_size": data_size, "trials": args.trials, "method": args.method}
    saveData["spectral_density_estimates"] = SDE(data, params)
    saver(saveData)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SDE eval variables")
    parser.add_argument('--dataset', '-d',
                        dest='dataset', 
                        type=str, 
                        default="random", 
                        required=False,
                        help="choose datasets here")
    parser.add_argument('--method', '-m',
                        dest='method', 
                        type=str, 
                        default="BlockKrylov", 
                        choices=["BlockKrylov", "SLQ"],
                        required=False, 
                        help="choose matvec method")
    parser.add_argument('--trials', '-t',
                        dest='trials', 
                        type=int, 
                        default=3, 
                        required=False,
                        help="number of trials to average out performance")
    parser.add_argument('--block_size', '-b',
                        dest='block_size', 
                        type=str, 
                        default="full", 
                        required=False,
                        help="block size for ther algos to run on")
    args = parser.parse_args()
    print(args)
    main(args)
