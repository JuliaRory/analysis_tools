import numpy as np
from scipy.linalg import eigh

def cov_epoch(X):
    """
    X: (channels, time)
    """
    C = X @ X.T
    return C / np.trace(C)

def regularize(C, alpha=0.05):
    return (1 - alpha) * C + alpha * np.eye(C.shape[0])

def calculate_CSP(epochs_motor, epochs_rest):
    covs_motor = np.array([cov_epoch(ep.T) for ep in epochs_motor])  # ep: (time, ch)
    C_motor  = covs_motor.mean(axis=0)
    C_motor  = regularize(C_motor,  alpha=0.05)

    covs_rest = np.array([cov_epoch(ep.T) for ep in epochs_rest])  # ep: (time, ch)
    C_rest  = covs_rest.mean(axis=0)
    C_rest = regularize(C_rest, alpha=0.05)
    
    C_sum = C_motor + C_rest
    eigvals, eigvecs = eigh(C_motor, C_sum)     # λ = 1 -> motor class
    
    # сортируем по убыванию собственных значений, первые - лучшие 
    ix = np.argsort(eigvals)[::-1]  # убывание λ
    eigvals = eigvals[ix]
    eigvecs = eigvecs[:, ix]

    ## spatial patterns 
    A = C_sum @ eigvecs
    A /= np.linalg.norm(A, axis=0, keepdims=True) # to normalize

    return eigvals, eigvecs, A