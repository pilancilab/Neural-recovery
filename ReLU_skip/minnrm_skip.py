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
    parser.add_argument("--seed", type=int, default=97006855, help="random seed")
    parser.add_argument("--sample", type=int, default=10, help="number of trials")
    parser.add_argument("--optw", type=int, default=1, help="choice of w")
    # 0: randomly generated (Gaussian) 1: smallest right eigenvector of X
    parser.add_argument("--optx", type=int, default=0, help="choice of X")
    # 0: Gaussian 1: cubic Gaussian
    # 2: 0 + whitened 3: 1 + whitened
    parser.add_argument("--sigma", type=float, default=0, help="noise")
    parser.add_argument("--save_details", type=bool, default=False, help="whether to save results of each convex program")
    parser.add_argument("--save_folder", type=str, default='./results/', help="path to save results")
    return parser


def gen_data(n, d):
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
        w = np.random.randn(d)
        w = w / np.linalg.norm(w)
    elif optw == 1:
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        w = Vh[-1, :].T

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

def solve_problem(n, d, sigma):
    data = {}  # empty dict
    X, w = gen_data(n, d)
    z = np.random.randn(n) * sigma / math.sqrt(n)
    y = X @ w + z
    data['X'] = X
    data['w'] = w
    data['y'] = y

    mh = max(n, 50)
    U1 = np.random.randn(d, mh)
    dmat = (X @ U1 >= 0)
    dmat, ind = np.unique(dmat, axis=1, return_index=True)
    if check(X):
        dmat = np.concatenate([dmat, np.ones((n, 1))], axis=1)
        data['exist_all_one'] = True
    else:
        data['exist_all_one'] = False


    # CVXPY variables
    m1 = dmat.shape[1]
    W0 = cp.Variable((d, ))
    W = cp.Variable((d, m1))
    obj = cp.norm(W0, 2) + cp.mixed_norm(W.T, 2, 1)
    constraints = [cp.sum(cp.multiply(dmat, (X @ W)), axis=1) + X @ W0 == y]
    # solve the problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    param_dict = {}
    prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params=param_dict)

    w0 = W0.value
    optw = W.value
    dis1 = np.linalg.norm(w - w0)
    X1, z = gen_data(n, d)
    y_predict = np.sum(np.maximum(0, X1 @ optw), axis=1) + X1 @ w0
    dis2 = np.linalg.norm(y_predict - X1 @ w)

    data['dmat'] = dmat
    data['opt_w0'] = w0
    data['opt_w'] = optw
    data['dis_abs'] = dis1
    data['dis_test'] = dis2
    return data


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(str(args))

    save_folder = args.save_folder
    seed = args.seed
    np.random.seed(seed)
    sigma = args.sigma
    sample = args.sample
    optw = args.optw
    optx = args.optx
    flag = args.save_details
    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)
    dlen = dvec.size
    nlen = nvec.size

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    dis_abs = np.zeros((nlen, dlen,sample))
    dis_test = np.zeros((nlen, dlen,sample))

    for nidx, n in enumerate(nvec):
        print('n = ' + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            for i in range(sample):
                data = solve_problem(n, d, sigma)
                dis_abs[nidx, didx, i] = data['dis_abs']
                dis_test[nidx, didx, i] = data['dis_test']
                if flag:
                    fname = 'minnrm_skip_n{}_d{}_w{}_X{}_sig{}_sample{}'.format(n, d, optw, optx, sigma, i)
                    file = open(save_folder + fname + '.pkl', 'wb')
                    pickle.dump(data, file)
                    file.close()

        t1 = time()
        print('time = ' + str(t1 - t0))

    fname = 'minnrm_skip_n{}_d{}_w{}_X{}_sig{}_sample{}'.format(args.n, args.d, optw, optx, sigma, sample)
    np.save(save_folder + 'dis_abs_' + fname, dis_abs)
    np.save(save_folder + 'dis_test_' + fname, dis_test)


if __name__ == '__main__':
    main()
