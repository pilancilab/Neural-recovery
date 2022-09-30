import argparse
import math
import os
import pickle
from time import time

import numpy as np
import torch
import torch.optim as optim
from relu_normal_nn import ReLUnormal

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
    parser.add_argument("--verbose", type=bool, default=False, help="whether to print information while training")
    parser.add_argument("--save_details", type=bool, default=False, help="whether to save training results")
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

def train_model(n, d, sigma, neu, verbose):
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
    Xtest, z = gen_data(n, d, neu)
    ytest = np.maximum(0, Xtest @ w)
    nrmy = np.linalg.norm(ytest, axis=0)
    ytest = np.sum(ytest / nrmy, axis=1)
    data['X_test'] = Xtest
    data['y_test'] = ytest
    y = y.reshape((n,1))
    ytest = ytest.reshape((n,1))

    Xtrain = torch.from_numpy(X).float()
    ytrain = torch.from_numpy(y).float()
    Xtest = torch.from_numpy(Xtest).float()
    ytest = torch.from_numpy(ytest).float()

    m = n + 1
    model = ReLUnormal(m=m, n=n, d=d)
    num_epoch = 400
    beta = 1e-6
    learning_rate = 2e-3
    loss_train = np.zeros(num_epoch)
    loss_test = np.zeros(num_epoch)

    if verbose:
        print('---------------------------training---------------------------')

    y_predict = model(Xtrain)
    loss = torch.linalg.norm(y_predict - ytrain) ** 2
    train_err_init = loss.item()
    with torch.no_grad():
        test_err_init = torch.linalg.norm(model(Xtest) - ytest) ** 2
        test_err_init = test_err_init.item()

    if verbose:
        print("Epoch [{}/{}], Train error: {}, Test error: {}".format(0, num_epoch, train_err_init,
                                                                      test_err_init))

    for epoch in range(num_epoch):
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_predict = model(Xtrain)
        loss = torch.linalg.norm(y_predict - ytrain) ** 2
        loss_train[epoch] = loss.item()
        with torch.no_grad():
            test_err = torch.linalg.norm(model(Xtest) - ytest) ** 2
            loss_test[epoch] = test_err.item()

        if verbose:
            print("Epoch [{}/{}], Train error: {}, Test error: {}".format(epoch + 1, num_epoch, loss_train[epoch],
                                                                          loss_test[epoch]))

    loss_train = np.concatenate([np.array([train_err_init]), loss_train])
    loss_test = np.concatenate([np.array([test_err_init]), loss_test])
    data['loss_train'] = loss_train
    data['loss_test'] = loss_test
    data['dis_test'] = math.sqrt(loss_test[-1])
    return data, model


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

    dis_test = np.zeros((nlen, dlen, sample))

    for nidx, n in enumerate(nvec):
        print('n = ' + str(n))
        t0 = time()
        for didx, d in enumerate(dvec):
            for i in range(sample):
                data, model = train_model(n, d, sigma, neu, verbose=args.verbose)

                dis_test[nidx, didx, i] = data['dis_test']
                if flag:
                    fname = '_n{}_d{}_w{}_X{}_sig{}_sample{}'.format(n, d, optw, optx, sigma, i)
                    file = open(save_folder + 'ncvx_train_normal' + fname + '.pkl', 'wb')
                    pickle.dump(data, file)
                    file.close()
                    torch.save(model.state_dict(), save_folder + model.name() + fname)

        t1 = time()
        print('time = ' + str(t1 - t0))

    fname = '_n{}_d{}_w{}_X{}_sig{}_sample{}'.format(args.n, args.d, optw, optx, sigma, sample)
    np.save(save_folder + 'dis_test_ncvx_train_normal' + fname, dis_test)


if __name__ == '__main__':
    main()
