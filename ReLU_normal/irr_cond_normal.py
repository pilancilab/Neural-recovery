import argparse
import math
import os
import pickle
from time import time

import cvxpy as cp
import numpy as np
import scipy.optimize as sciopt

def get_parser():
    parser = argparse.ArgumentParser(description="phase transition")
    parser.add_argument("--n", type=int, default=400, help="number of sample")
    parser.add_argument("--d", type=int, default=100, help="number of dimension")
    parser.add_argument("--neu", type=int, default=2, help="number of planted neuron")
    parser.add_argument("--seed", type=int, default=97006855, help="random seed")
    parser.add_argument("--sample", type=int, default=50, help="number of trials")
    parser.add_argument("--optw", type=int, default=0, help="choice of w")
    # 0: randomly generated (Gaussian)
    # 1: u_i = e_i
    # 2: (only for 2 neurons) u_1 = u, u_2 = - u
    parser.add_argument("--optx", type=int, default=0, help="choice of X")
    # 0: Gaussian 1: cubic Gaussian
    # 2: 0 + whitened 3: 1 + whitened
    parser.add_argument("--save_folder", type=str, default='./results/', help="path to save results")
    return parser


def gen_data(n, d, neu):
    parser = get_parser()
    args = parser.parse_args()
    optx = args.optx
    optw = args.optw

    X = np.random.randn(n, d) / math.sqrt(n)
    if optx in [1, 3]:
        X = X ** 3
    if optx in [2, 4]:
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        if n < d:
            X = Vh
        else:
            X = U

    if optw == 0:
        w = np.random.randn(d, neu)
    elif optw == 1:
        w = np.eye(d, neu)
    elif optw == 2:
        if neu == 2:
            w = np.random.randn(d, 1)
            w = np.concatenate([w, -w],axis=1)
        else:
            raise TypeError("Invalid choice of planted neurons.")

    return X, w

def check(X):
    n, d = X.shape
    nrm = lambda x: np.linalg.norm(x, 2)
    x0 = np.random.randn(d)
    x0 = x0 / np.linalg.norm(x0)
    lc = sciopt.LinearConstraint(X, 0, np.inf)
    nlc = sciopt.NonlinearConstraint(nrm, 0, 1)
    res = sciopt.minimize(lambda x: - nrm(x), x0, constraints=[lc, nlc])
    if -res.fun <= 1e-6:
        return False # no all-one arrangement
    else:
        return True # exist all-one arrangement

def check_irr(n, d, neu):
    eps = 1e-10
    while True:
        X, w = gen_data(n, d, neu)
        nrmw = np.linalg.norm(w, axis=0)
        w = w / nrmw
        y = np.maximum(0, X @ w)
        nrmy = np.linalg.norm(y, axis=0)
        if np.all(nrmy >= eps):
            break

    mh = max(n * 2, 50)
    U1 = np.concatenate([w,np.random.randn(d, mh)],axis=1)
    dmat = (X @ U1 >= 0)
    dmat, ind = np.unique(dmat, axis=1, return_index=True)
    j_array = np.nonzero(ind <= neu - 1)[0]
    j_map = ind[j_array]
    if check(X):
        dmat = np.concatenate([dmat,np.ones((n,1))],axis=1)

    U = np.zeros((n,0))
    uu = []
    for jidx, j in enumerate(j_array):
        k = j_map[jidx]
        Xj = dmat[:, j].reshape((n, 1)) * X
        Uj, Sj, Vjh = np.linalg.svd(Xj, full_matrices=False)
        rj = np.linalg.matrix_rank(Xj)
        wj = (Sj.reshape((d, 1)) * Vjh) @ w[:, k]
        wj = wj / np.linalg.norm(wj)
        U = np.concatenate([U, Uj[:,np.arange(rj)]], axis=1)
        uu = np.concatenate([uu, wj[np.arange(rj)]])
    lam = U @ np.linalg.pinv(U.T @ U) @ uu

    m1 = dmat.shape[1]
    count = 0
    for j in range(m1):
        if j in j_array:
            continue
        dj = dmat[:, j]
        Xj = dj.reshape((n, 1)) * X
        Uj, Sj, Vjh = np.linalg.svd(Xj, full_matrices=False)
        if np.linalg.norm(Uj.T @ lam) >= 1 + eps:
            count += 1

    return (count == 0)


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(str(args))

    save_folder = args.save_folder
    seed = args.seed
    np.random.seed(seed)
    sample = args.sample
    optw = args.optw
    optx = args.optx
    neu = args.neu
    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)
    dlen = dvec.size
    nlen = nvec.size

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    prob = np.zeros((nlen, dlen, sample))

    for nidx, n in enumerate(nvec):
        print('n = ' + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            if n < d:
                prob[nidx, didx, :] = False
                continue
            for i in range(sample):
                prob[nidx, didx, i] = check_irr(n, d, neu)

        t1 = time()
        print('time = ' + str(t1 - t0))

    fname = 'irr_cond_n{}_d{}_w{}_X{}_sample{}'.format(args.n, args.d, optw, optx, sample)
    np.save(save_folder + fname, prob)
    print(np.mean(prob,axis=2))


if __name__ == '__main__':
    main()
