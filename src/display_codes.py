import matplotlib.pyplot as plt
import os

def plotter(errors, standard_deviations, blocks, methods, dataset):
    plt.rcParams.update({'font.size': 13})
    
    label_names = {"SLQ": "SLQ", "BKDE": "Krylov deflation + SDE"}
    
    for m in methods:
        plt.plot(blocks[m], errors[m], label = label_names[m])
        plt.fill_between(blocks[m], \
                        errors[m] - standard_deviations[m],  \
                        errors[m] + standard_deviations[m])
    plt.legend()
    plt.xlabel("Block size as log proportion of dataset")
    plt.ylabel("Wasserstein error of SDE")
    plt.title(dataset)
    
    saveDestination = os.path.join("figures", dataset)
    if not os.path.isdir(saveDestination):
        os.makedirs(saveDestination)
    methods_names = ""
    for m in methods:
        if methods_names == "":
            methods_names = m
        else:
            methods_names += "_"+m
    
    saveFile = os.path.join(saveDestination, methods_names+".pdf")
    plt.savefig(saveFile)
    return None