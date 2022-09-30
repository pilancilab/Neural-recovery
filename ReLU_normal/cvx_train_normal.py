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
    parser.add_argument("--sample", type=int, default=10, help="number of trials")
    parser.add_argument("--optw", type=int, default=0, help="choice of w")
    # 0: randomly generated (Gaussian)
    # 1: u_i = e_i
    # 2: (only for 2 neurons) u_1 = u, u_2 = - u
    parser.add_argument("--optx", type=int, default=0, help="choice of X")
    # 0: Gaussian 1: cubic Gaussian
    # 2: 0 + whitened 3: 1 + whitened
    parser.add_argument("--sigma", type=float, default=0, help="noise")
    parser.add_argument("--save_details", type=bool, default=False, help="whether to save results of each convex program")
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

def solve_problem(n, d, sigma, neu):
    data = {}  # empty dict
    while True:
        X, w = gen_data(n, d, neu)
        nrmw = np.linalg.norm(w, axis=0)
        w = w / nrmw
        y = np.maximum(0, X @ w)
        nrmy = np.linalg.norm(y, axis=0)
        if np.all(nrmy >= 1e-10):
            break
    y = np.sum(y / nrmy, axis=1)
    z = np.random.randn(n) * sigma / math.sqrt(n)
    y = y + z
    data['X'] = X
    data['w'] = w
    data['y'] = y

    mh = max(n, 50)
    U1 = np.concatenate([w, np.random.randn(d, mh)],axis=1)
    dmat = (X @ U1 >= 0)
    dmat, ind = np.unique(dmat, axis=1, return_index=True)
    if check(X):
        dmat = np.concatenate([dmat, np.ones((n, 1))], axis=1)
        data['exist_all_one'] = True
    else:
        data['exist_all_one'] = False

    # CVXPY variables
    m1 = dmat.shape[1]
    W1 = cp.Variable((d, m1))
    W2 = cp.Variable((d, m1))
    expr = np.zeros(n)
    constraints = []
    for i in range(m1):
        di = dmat[:, i].reshape((n, 1))
        Xi = di * X
        Ui, S, Vh = np.linalg.svd(Xi, full_matrices=False)
        ri = np.linalg.matrix_rank(Xi)
        if ri == 0:
            constraints += [W1[:, i] == 0, W2[:, i] == 0]
        else:
            expr += Ui[:, np.arange(ri)] @ (W1[np.arange(ri), i] - W2[np.arange(ri), i])
            X1 = X @ Vh[np.arange(ri), :].T @ np.diag(1 / S[np.arange(ri)])
            X2 = (2 * di - 1) * X1
            constraints += [X2 @ W1[np.arange(ri), i] >= 0, X2 @ W2[np.arange(ri), i] >= 0]
            if ri < d:
                constraints += [W1[np.arange(ri, d), i] == 0, W2[np.arange(ri, d), i] == 0]
    loss = cp.norm(expr - y, 2) ** 2
    regw = cp.mixed_norm(W1.T, 2, 1) + cp.mixed_norm(W2.T, 2, 1)
    beta = 1e-6
    obj = loss + beta * regw
    # solve the problem
    prob = cp.Problem(cp.Minimize(obj), constraints)
    param_dict = {}
    prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False, mosek_params=param_dict)

    w1 = W1.value
    w2 = W2.value
    data['i_map'] = np.zeros(neu)
    sum_square = 0
    for j in range(neu):
        k = np.nonzero(ind == j)[0][0]
        data['i_map'][j] = k
        wj = w[:, j]
        dj = dmat[:, k]
        Xj = dj.reshape((n, 1)) * X
        Uj, Sj, Vjh = np.linalg.svd(Xj, full_matrices=False)
        wj = (Sj.reshape((d, 1)) * Vjh) @ wj
        wj = wj / np.linalg.norm(wj)
        sum_square += np.linalg.norm(w1[:, k] - w2[:, k] - wj) ** 2
    dis1 = math.sqrt(sum_square)

    data['dmat'] = dmat
    data['opt_w1'] = w1
    data['opt_w2'] = w2
    data['dis_abs'] = dis1
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
    neu = args.neu
    flag = args.save_details
    dvec = np.arange(10, args.d + 1, 10)
    nvec = np.arange(10, args.n + 1, 10)
    dlen = dvec.size
    nlen = nvec.size

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    dis_abs = np.zeros((nlen, dlen, sample))

    for nidx, n in enumerate(nvec):
        print('n = ' + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            if n < d:
                dis_abs[nidx, didx, :] = None
                continue
            for i in range(sample):
                data = solve_problem(n, d, sigma, neu)
                dis_abs[nidx, didx, i] = data['dis_abs']
                if flag:
                    fname = 'cvx_train_normal_n{}_d{}_w{}_X{}_sig{}_sample{}'.format(n, d, optw, optx, sigma, i)
                    file = open(save_folder + fname + '.pkl', 'wb')
                    pickle.dump(data, file)
                    file.close()

        t1 = time()
        print('time = ' + str(t1 - t0))

    fname = 'cvx_train_normal_n{}_d{}_w{}_X{}_sig{}_sample{}'.format(args.n, args.d, optw, optx, sigma, sample)
    np.save(save_folder + 'dis_abs_' + fname, dis_abs)


if __name__ == '__main__':
    main()
