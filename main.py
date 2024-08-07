import numpy as np
import os, sys
import pickle
from src.get_dataset import get_data
from src.utils import get_spectrum, saver
from src.approx_wrapper import checkSDEApproxError
import matplotlib.pyplot as plt
import matplotlib

def map_name(name):
    if name == "gaussian":
        return "Gaussian matrix"
    if name == "low_rank_matrix":
        return "Low-rank matrix"
    if name == "power_law_spectrum":
        return "Power law spectrum matrix"
    if name == "uniform":
        return "Uniform matrix"
    if name == "inverse_spectrum":
        return "Inverse spectrum matrix"
    if name == "erdos992":
        return "Erdos992 adjacency matrix"
    return None

def main(random_restarts=5, dataset_names = "all", methods = ["all"], loadresults = [True, True, True, True, True, True], variation="fixed"):
    matplotlib.rcParams.update({'font.size': 16})
    # colors chosen from https://matplotlib.org/stable/gallery/color/named_colors.html
    colors = ["red", "dodgerblue", "black", "darkorchid", "#D2691E", "#40E0D0"]
    if dataset_names == "all":
        ds = ["gaussian", "uniform", "low_rank_matrix", "power_law_spectrum", "inverse_spectrum", "erdos992"] # "hypercube", "gaussian", "uniform", "erdos992", "small_large_diagonal", "square_inverse_spectrum"
    else:
        ds = [dataset_names]
    if methods[-1] == "all":
        methods = ["SLQMM", "CMM", "KPM", "VRSLQMM-c1", "VRSLQMM-c2", "VRSLQMM-c12", "BKSDE-CMM", "BKSDE-KPM"]
    else:
        pass
    if len(loadresults) != len(methods):
        print("loadresults should be of same size")
        sys.exit(1)        
    for dataset in ds:
        print("running for dataset:", dataset)
        print("random restarts:", random_restarts)
        # dataset = "hypercube"
        load_mat_flag = True
        data, n = get_data(dataset, load=load_mat_flag)
        if np.linalg.norm(data, ord=2) > 1:
            data /= np.linalg.norm(data, ord=2)
        eigs_folder = "outputs/"+dataset+"/"+"_"+variation+"/"
        if not os.path.isdir(eigs_folder):
            os.makedirs(eigs_folder)
        eigs_file = eigs_folder+"true_eigvals.npy"
        if os.path.isfile(eigs_file) and load_mat_flag:
            support_true = np.load(eigs_file)
        else:
            support_true = np.real(np.linalg.eigvals(data))
            np.save(eigs_file, support_true)
        # set up moments ###################################### MOMENT VALS
        moments = np.arange(8,60,4, dtype=int)
        
        foldername = "outputs/"+dataset+"/"+str(random_restarts)+"_"+variation+"/"
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
        
        # setting up figure
        plt.gcf().clf()
        fig = plt.figure()
        ax = fig.add_subplot()
        
        # run the full code
        for i in range(len(methods)):
            print(methods[i])
            # set up file name
            filename = foldername+"/"+methods[i]+".pkl"
            # check if file with results exist, if yes load, else run code
            if os.path.isfile(filename) and loadresults[i] == True:
                file_ = open(filename, "rb")
                errors_mean, errors_lo, errors_hi = pickle.load(file_)
                file_.close()
            else:
                errors_mean, errors_lo, errors_hi = checkSDEApproxError(data, moments, support_true, method=methods[i], cheb_vals=15000, random_restarts=random_restarts,variation=variation)
                # save results to filename
                file_ = open(filename, "wb")
                pickle.dump([errors_mean, errors_lo, errors_hi], file_)
                file_.close()
            
            # fixing name for VRSLQ
            if "VRSLQMM" in methods[i]:
                methods[i] = "VR-SLQ"
            if "BKSDE-CMM" in methods[i]:
                methods[i] = "def-CMM"
            if "BKSDE-KPM" in methods[i]:
                methods[i] = "def-KPM"
            if "SLQMM" in methods[i]:
                methods[i] = "SLQ"
            # plot errors with low and high
            ax.plot(random_restarts*moments, errors_mean, label=methods[i], color=colors[i])
            ax.fill_between(random_restarts*moments, errors_lo, errors_hi, alpha=0.2, color=colors[i])
            
        # plt.legend()
        handles,labels = ax.get_legend_handles_labels()
        plt.ylabel("Wasserstein error")
        plt.yscale("log")
        plt.xlabel("Total matrix-vector queries")
        # plt.yticks([10**0, 10**(-1), 10**(-2), 10**(-3)])
        plt.yticks([10**(-1), 10**(-2)])
        plt.grid()
        plt.title(map_name(dataset))
        if not os.path.isdir("figures/unittests/SDE_approximation_errors/"+str(random_restarts)+"_"+variation+"/"):
            os.makedirs("figures/unittests/SDE_approximation_errors/"+str(random_restarts)+"_"+variation+"/")
        plt.savefig("figures/unittests/SDE_approximation_errors/"+str(random_restarts)+"_"+variation+"/"+dataset+".pdf", bbox_inches='tight', dpi=200)
        # plt.clf()
        # plt.close()
        
        # save legend in a a separate file
        plt.gcf().clf()
        fig_legend = plt.figure()
        leg = fig_legend.legend(handles, labels, ncol=6)
        leg_lines = leg.get_lines()
        plt.setp(leg_lines, linewidth=2)
        fig_legend.savefig("figures/unittests/SDE_approximation_errors/"+str(random_restarts)+"_"+variation+"/"+"legend.pdf", bbox_inches='tight')
        plt.gcf().clf()
    return None

if __name__ == "__main__":
    try:
        val = int(sys.argv[1])
    except:
        val = 15
    try:
        var = sys.argv[2]
    except:
        var = "fixed"
    
    mults = [val]
    dataset_names = "gaussian"
    methods = ["SLQMM", "CMM", "KPM", "VRSLQMM-c12", "BKSDE-CMM", "BKSDE-KPM"] # ["SLQMM", "CMM", "KPM", "VRSLQMM-c12", "BKSDE-CMM", "BKSDE-KPM"]
    loadresults = [True, True, True, True, True, True] # [True, True, True, True, False, True]
    for mult in mults:
        print("###################### random restarts:", mult)
        main(mult, dataset_names, methods, loadresults, variation=var)
