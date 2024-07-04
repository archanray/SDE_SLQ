import numpy as np
from tqdm import tqdm
from src.moment_estimator import approxChebMomentMatching, discretizedJacksonDampedKPM, hutchMomentEstimator, SLQMM, VRSLQMM, adder, bkde
import scipy as sp

def sdeComputer(data, degree, method = "CMM", cheb_vals=None, submethod="cvxpy", eigvals=None, random_restarts=1, rand_vects=None):
    if method == "CMM":
        tau = hutchMomentEstimator(data, degree, random_restarts, G=rand_vects)
        supports, q = approxChebMomentMatching(tau, method=submethod, cheb_vals=cheb_vals)
        # print(supports, q)
        return supports, q
    if method == "KPM":
        tau = hutchMomentEstimator(data, degree, random_restarts, G=rand_vects)
        supports, q = discretizedJacksonDampedKPM(tau)
        return supports, q
    if method == "SLQMM":
        return SLQMM(data, degree, random_restarts, V=rand_vects)
    if method == "VRSLQMM-c12" or method == "VRSLQMM":
        return VRSLQMM(data, degree, random_restarts, constraints="12", V=rand_vects)
    if method == "VRSLQMM-c1":
        return VRSLQMM(data, degree, random_restarts, constraints="1", V=rand_vects)
    if method == "VRSLQMM-c2":
        return VRSLQMM(data, degree, random_restarts, constraints="2", V=rand_vects)
    if method == "BKSDE-CMM":
        return bkde(data, degree, random_restarts, MM="CMM", cheb_vals=cheb_vals)
    if method == "BKSDE-KPM":
        return bkde(data, degree, random_restarts, MM="KPM")
    return None

def checkSDEApproxError(data, moments, support_true, method="CMM", cheb_vals=1000, trials=5, submethod="cvxpy", random_restarts=1, q_lo=20, q_hi=80, variation="fixed"):
    quantile_lo = q_lo
    quantile_hi = q_hi
    errors = np.zeros((trials,len(moments)))
    pdf_true = np.ones_like(support_true) / len(support_true)
    eigvals = np.real(np.linalg.eigvals(data))
    
    # colors = sns.color_palette('hls', len(moments))
    
    for t in tqdm(range(trials)):
        # generate random vectors
        n = len(data)
        l = random_restarts
        if variation == "fixed":
            rand_vects = np.random.normal(loc=0.,scale=1., size=(n, l)) #####################CHANGE THIS IN THE MORNING TO RUN ANOTHER
        else:
            rand_vects = None
        for j in range(len(moments)):
            support_current, pdf_current = sdeComputer(data, moments[j], method = method, cheb_vals = cheb_vals, submethod=submethod, eigvals=eigvals, random_restarts=random_restarts, rand_vects=rand_vects)
            # if j == len(moments)-1:
                # axs[0].scatter(support_current, pdf_current, label=str(moments[j]))
            errors[t,j] = sp.stats.wasserstein_distance(support_true, support_current, pdf_true, pdf_current)
            # errors[t,j] = sp.stats.wasserstein_distance(support_true, support_current)
        pass
        # axs[1].plot(moments, errors[t,:])
        # axs[0].legend()
        # plt.savefig("figures/unittests/"+method+"_pdf_and_errors_at_moment_only_support.pdf", bbox_inches='tight', dpi=200)
    
    errors_mean = np.mean(errors, axis=0)
    errors_lo = np.percentile(errors, q=quantile_lo, axis=0)
    errors_hi = np.percentile(errors, q=quantile_hi, axis=0)
    
    return errors_mean, errors_lo, errors_hi
