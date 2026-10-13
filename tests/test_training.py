"""
Tests of the training workflow used in notebooks/ts_classification.ipynb with the TF2/GPflow2 port
(the optimizers differ from TF1, so trajectories are not compared against the original code).
Run from the repository root with `python -m pytest tests`.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gpflow
import gpsig


def make_data(num_examples, len_examples=8, num_features=2, num_classes=3, seed=0):
    # each class is a noisy random walk with a class-specific drift
    rng = np.random.RandomState(seed)
    y = rng.randint(0, num_classes, size=num_examples)
    drift = np.stack([np.cos(2 * np.pi * y / num_classes), np.sin(2 * np.pi * y / num_classes)], axis=1)
    X = (drift[:, None, :] + 0.5 * rng.randn(num_examples, len_examples, num_features)).cumsum(axis=1) / len_examples
    # same preprocessing as in the notebook
    X = gpsig.preprocessing.add_time_to_list(list(X))
    return gpsig.preprocessing.tabulate_list_of_sequences(X), y


@pytest.fixture(scope='module')
def setup():
    np.random.seed(0)
    num_levels, num_inducing, num_classes = 3, 12, 3
    X_train, y_train = make_data(60, seed=0)
    X_val, y_val = make_data(30, seed=1)
    num_train, len_examples, num_features = X_train.shape
    Z_init = gpsig.utils.suggest_initial_inducing_tensors(X_train, num_levels, num_inducing, labels=y_train, increments=True)
    l_init = gpsig.utils.suggest_initial_lengthscales(X_train, num_samples=100)
    input_dim = len_examples * num_features
    X_train, X_val = X_train.reshape([-1, input_dim]), X_val.reshape([-1, input_dim])
    feat = gpsig.inducing_variables.InducingTensors(Z_init, num_levels=num_levels, increments=True)
    k = gpsig.kernels.SignatureRBF(input_dim, num_features, num_levels, lengthscales=l_init)
    lik = gpflow.likelihoods.MultiClass(num_classes)
    m = gpsig.models.SVGP(X_train, y_train[:, None], kern=k, feat=feat, likelihood=lik, num_latent=num_classes, minibatch_size=20)
    acc = lambda m, X, y: np.mean(y == np.argmax(m.predict_y(X)[0], axis=1))
    nlpp = lambda m, X, y: -np.mean(m.predict_density(X, y[:, None]))
    return dict(m=m, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                val_acc=lambda m: acc(m, X_val, y_val), val_nlpp=lambda m: nlpp(m, X_val, y_val))


def test_notebook_workflow(setup):
    m, val_acc, val_nlpp = setup['m'], setup['val_acc'], setup['val_nlpp']
    opt = gpsig.training.NadamOptimizer
    elbo_init = m.compute_log_likelihood()
    nlpp_init = val_nlpp(m)

    # 1. fixed kernel hyperparameters
    m.kern.set_trainable(False)
    kern_values = {k: v.numpy() for k, v in gpflow.utilities.parameter_dict(m.kern).items()}
    hist = gpsig.training.optimize(m, opt(1e-2), max_iter=100, print_freq=10, save_freq=20, val_scorer=[val_acc, val_nlpp])
    for k, v in gpflow.utilities.parameter_dict(m.kern).items():
        np.testing.assert_array_equal(v.numpy(), kern_values[k])
    assert sorted(x for x in hist if str(x).isnumeric()) == [20, 40, 60, 80, 100]
    assert all(set(hist[i]) == {'time', 'elbo', 'val'} and len(hist[i]['val']) == 2 for i in [20, 100])
    assert m.compute_log_likelihood() > elbo_init

    # 2. unfixed kernel, continuing the history, with early stopping on the validation nlpp
    m.kern.set_trainable(True)
    hist = gpsig.training.optimize(m, opt(1e-2), max_iter=200, print_freq=10, save_freq=20, history=hist,
                                   val_scorer=[val_acc, val_nlpp], save_best_params=True, lower_is_better=True, patience=1000)
    iters = sorted(x for x in hist if str(x).isnumeric())
    assert iters[-1] == 300 and hist[300]['time'] > hist[100]['time']
    assert any(not np.allclose(v.numpy(), kern_values[k]) for k, v in gpflow.utilities.parameter_dict(m.kern).items())
    best = hist['best']
    assert best['val'][-1] == min(hist[i]['val'][-1] for i in iters if i > 100)
    m.assign(best['params'])
    assert np.isclose(val_nlpp(m), best['val'][-1])
    assert val_nlpp(m) < nlpp_init

    # 3. merge the validation data into the training data
    X_all = np.concatenate((setup['X_train'], setup['X_val']), axis=0)
    y_all = np.concatenate((setup['y_train'], setup['y_val']), axis=0)
    m.X, m.Y = X_all, y_all  # 1-d labels, as in the notebook
    m.num_data = X_all.shape[0]
    m.kern.set_trainable(False)
    hist = gpsig.training.optimize(m, opt(1e-3), max_iter=40, print_freq=10, save_freq=20, history=hist)
    assert sorted(x for x in hist if str(x).isnumeric())[-1] == 340
    assert np.isfinite(m.compute_log_likelihood())


def test_early_stopping(setup):
    m = setup['m']
    # with a validation score that only gets worse, training stops after `patience` iterations without improvement
    scores = iter(range(1000))
    hist = gpsig.training.optimize(m, gpsig.training.AdamOptimizer(1e-3), max_iter=200, save_freq=10,
                                   val_scorer=lambda m: float(next(scores)), save_best_params=True, lower_is_better=True, patience=30)
    iters = sorted(x for x in hist if str(x).isnumeric())
    assert hist['best']['iter'] == 10 and iters[-1] == 50


def test_multiple_optimizers(setup):
    m = setup['m']
    gpflow.set_trainable(m, True)
    Z_before, sigma_before = m.feature.Z.numpy(), m.kern.sigma.numpy()
    q_mu_before = m.q_mu.numpy()
    # first optimizer on the inducing tensors only, second (zero learning rate) on all remaining variables
    opts = [gpsig.training.AdamOptimizer(1e-2), gpsig.training.GradientDescentOptimizer(0.)]
    gpsig.training.optimize(m, opts, var_list=[[m.feature.Z]], max_iter=10, save_freq=10)
    assert not np.allclose(m.feature.Z.numpy(), Z_before)
    np.testing.assert_array_equal(m.kern.sigma.numpy(), sigma_before)
    np.testing.assert_array_equal(m.q_mu.numpy(), q_mu_before)
