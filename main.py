import numpy as np
import argparse
from src.get_dataset import get_data
from src.utils import get_spectrum, saver
from src.approx_wrapper import SDE

def main(args):
    data, data_size = get_data(args.dataset)
    true_spectrum_sorted = get_spectrum(data)
    block_sizes = [x for x in range(10, int(np.sqrt(data_size)), 50)]
    saveData = {"true_spectrum": true_spectrum_sorted, \
                "dataset": args.dataset, \
                "trials": args.trials,\
                "method": args.method,\
                "block_sizes": block_sizes,\
                "iters": args.iters}
    params = {"data_size": data_size, \
                "trials": args.trials, \
                "method": args.method, \
                "block_sizes": block_sizes, \
                "iters": args.iters}
    saveData["spectral_density_estimates"], saveData["random_seeds"] = SDE(data, params)
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
    parser.add_argument('--iters', '-l',
                        dest='iters', 
                        type=int, 
                        default=10, 
                        required=False,
                        help="number of iterations in the estimators")
    args = parser.parse_args()
    print(args)
    main(args)
