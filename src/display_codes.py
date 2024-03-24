import matplotlib.pyplot as plt
import os

def plotter(errors, standard_deviations, blocks, methods, dataset):
    plt.rcParams.update({'font.size': 22})
    for m in methods:
        plt.plot(blocks, errors[m], label = m)
        plt.fill_between(blocks, \
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
    methods_names = [methods_names+"_"+x for x in methods]
    saveFile = os.path.join(saveDestination, methods_names+".pdf")
    plt.savefig(saveFile)
    return None