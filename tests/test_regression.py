"""
Regression tests of the TF2/GPflow2 port against outputs recorded with the original TF1/GPflow1 code
(see tests/reference/make_reference.py). Run from the repository root with `python -m pytest tests`.
"""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests', 'reference'))

import gpflow
import gpsig
import cases as C

REFERENCE = np.load(os.path.join(ROOT, 'tests', 'reference', 'gpsig_tf1_reference.npz'))
RTOL, ATOL = 1e-7, 1e-9


def assert_close(actual, desired, what):
    actual, desired = np.asarray(actual), np.asarray(desired)
    np.testing.assert_allclose(actual.reshape(desired.shape), desired, rtol=RTOL, atol=ATOL, err_msg=what)


def make_kernel(case, data):
    kern = getattr(gpsig.kernels, case['kern'])(data['input_dim'], C.NUM_FEATURES, C.NUM_LEVELS, **case['kwargs'])
    for name, value in C.kernel_param_values(case).items():
        getattr(kern, name).assign(value)
    return kern


def kernel_outputs(kern, d):
    X, Z, Z_incr, Zs = (tf.constant(d[k]) for k in ['X', 'Z', 'Z_incr', 'Zs'])
    fns = {
        'K_XX': lambda: kern.compute_K_symm(d['X']),
        'K_XX2': lambda: kern.compute_K(d['X'], d['X2']),
        'K_lvls_XX2': lambda: kern.compute_K_levels(d['X'], d['X2']),
        'Kdiag': lambda: kern.compute_Kdiag(d['X']),
        'Kdiag_lvls': lambda: kern.compute_K_level_diags(d['X']),
        'base_kern_XX': lambda: kern.compute_base_kern_symm(d['X']),
        'K_tens': lambda: kern.compute_K_tens(d['Z']),
        'K_incr_tens': lambda: kern.compute_K_incr_tens(d['Z_incr']),
        'K_tens_vs_seq': lambda: kern.compute_K_tens_vs_seq(d['Z'], d['X']),
        'K_incr_tens_vs_seq': lambda: kern.compute_K_incr_tens_vs_seq(d['Z_incr'], d['X']),
        'tens_covs': lambda: kern.K_tens_n_seq_covs(Z, X),
        'tens_covs_full': lambda: kern.K_tens_n_seq_covs(Z, X, full_X_cov=True),
        'incr_tens_covs': lambda: kern.K_tens_n_seq_covs(Z_incr, X, increments=True),
        'seq_covs': lambda: kern.K_seq_n_seq_covs(Zs, X),
    }
    return fns


@pytest.mark.parametrize('case', C.kernel_cases(), ids=lambda c: c['name'])
def test_kernel(case):
    d = C.kernel_data(case)
    if case['kern'] == 'SignatureSpectral':
        # disabled in the port, since it is broken in the original code (most outputs failed when recording the reference)
        with pytest.raises(NotImplementedError):
            make_kernel(case, d)
        return
    kern = make_kernel(case, d)
    prefix = 'kern/{}/'.format(case['name'])
    failed_in_original = set(REFERENCE[prefix + '_failed'].tolist())
    checked = 0
    for key, fn in kernel_outputs(kern, d).items():
        keys = [key] if key + '_Kzz' not in C.KERNEL_OUTPUTS else [key + s for s in ['_Kzz', '_Kzx', '_Kxx']]
        if all(k in failed_in_original for k in keys):
            continue
        out = fn()
        out = [out] if len(keys) == 1 else [np.asarray(o) for o in out]
        for k, o in zip(keys, out):
            assert_close(o, REFERENCE[prefix + k], '{} / {}'.format(case['name'], k))
            checked += 1
    assert checked > 0


@pytest.mark.parametrize('case', [c for c in C.kernel_cases() if c['kern'] != 'SignatureSpectral'], ids=lambda c: c['name'])
def test_seq_covs_consistent_with_K(case):
    # K_seq_n_seq_covs (used with InducingSequences) must give the same cross-covariance as K; with normalization,
    # the original code divided it by the square roots of the diagonal of Kzz twice (fixed in the port)
    d = C.kernel_data(case)
    kern = make_kernel(case, d)
    for return_levels in [False, True]:
        Kzz, Kzx, _ = kern.K_seq_n_seq_covs(tf.constant(d['Zs']), tf.constant(d['X']), return_levels=return_levels)
        np.testing.assert_allclose(Kzx, kern.K(d['Zs'], d['X'], presliced_X=True, return_levels=return_levels), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(Kzz, kern.K(d['Zs'], presliced=True, return_levels=return_levels), rtol=RTOL, atol=ATOL)


def test_low_rank_disabled():
    with pytest.raises(NotImplementedError):
        gpsig.kernels.SignatureRBF(12, C.NUM_FEATURES, C.NUM_LEVELS, low_rank=True)
    # low_rank=False remains the default
    gpsig.kernels.SignatureRBF(12, C.NUM_FEATURES, C.NUM_LEVELS, low_rank=False)


def make_model(case, d):
    kern = gpsig.kernels.SignatureRBF(d['input_dim'], C.NUM_FEATURES, C.NUM_LEVELS, num_lags=case['num_lags'])
    for name, value in d['kern_values'].items():
        getattr(kern, name).assign(value)
    feat = getattr(gpsig.inducing_variables, case['feat'])(d['Z'], num_levels=C.NUM_LEVELS, **case['feat_kwargs'])
    if case['lik'] == 'Gaussian':
        lik = gpflow.likelihoods.Gaussian(variance=d['lik_variance'])
    elif case['lik'] == 'Bernoulli':
        lik = gpflow.likelihoods.Bernoulli()
    else:
        lik = gpflow.likelihoods.MultiClass(C.NUM_CLASSES)
    m = gpsig.models.SVGP(d['X'], d['Y'], kern=kern, feat=feat, likelihood=lik, num_latent=d['num_latent'], whiten=case['whiten'])
    m.q_mu.assign(d['q_mu'])
    m.q_sqrt.assign(d['q_sqrt'])
    if case['feat_kwargs'].get('learn_weights'):
        m.feature.W.assign(d['W'])
    return m


def constrained_gradients(m, case, d):
    """
    Gradients of the ELBO w.r.t. the constrained parameter values (as recorded from GPflow 1).
    """
    params = {'variances': m.kern.variances, 'sigma': m.kern.sigma, 'lengthscales': m.kern.lengthscales,
              'Z': m.feature.Z, 'q_mu': m.q_mu, 'q_sqrt': m.q_sqrt}
    if case['num_lags']:
        params.update(lags=m.kern.lags, gamma=m.kern.gamma)
    if case['feat_kwargs'].get('learn_weights'):
        params['W'] = m.feature.W
    names = sorted(params)
    variables = [params[n].unconstrained_variable for n in names]
    with tf.GradientTape() as tape:
        elbo = m.elbo((tf.constant(d['X']), tf.constant(d['Y'])))
    grads = {}
    for n, g in zip(names, tape.gradient(elbo, variables)):
        p = params[n]
        if p.transform is None:
            grads[n] = g.numpy()
        elif n == 'q_sqrt':
            # FillTriangular only rearranges entries: unconstrained grads are the lower-triangular constrained grads
            grads[n] = p.transform.forward(g).numpy()
        else:
            # elementwise transforms: divide by the derivative of the transform
            u = p.unconstrained_variable
            with tf.GradientTape() as t:
                y = p.transform.forward(u)
            grads[n] = (g / t.gradient(y, u)).numpy()
    return grads


@pytest.mark.parametrize('case', C.model_cases(), ids=lambda c: c['name'])
def test_model(case):
    d = C.model_data(case)
    m = make_model(case, d)
    prefix = 'model/{}/'.format(case['name'])
    out = {'elbo': m.compute_log_likelihood()}
    out['f_mean'], out['f_var'] = m.predict_f(d['X_test'])
    out['y_mean'], out['y_var'] = m.predict_y(d['X_test'])
    out['log_density'] = m.predict_density(d['X_test'], d['Y_test'])
    for name, g in constrained_gradients(m, case, d).items():
        out['grad_' + name] = g
    assert sorted(k for k in REFERENCE.files if k.startswith(prefix)) == sorted(prefix + k for k in out)
    for key, value in out.items():
        if case['lik'] == 'MultiClass' and key not in ['f_mean', 'f_var']:
            # known difference in GPflow itself, not in gpsig: GPflow 2 squashes the CDFs in RobustMax.prob_is_largest
            # with 1e-6 instead of 1e-4 (GPflow 1), which changes everything computed by the MultiClass likelihood
            # (ELBO, predictive mean/var/density, gradients) by ~1e-4 relative; the latent f_mean, f_var are checked exactly
            desired = REFERENCE[prefix + key]
            np.testing.assert_allclose(np.asarray(value).reshape(desired.shape), desired, rtol=1e-3, atol=1e-3 * np.abs(desired).max(),
                                       err_msg='{} / {}'.format(case['name'], key))
            continue
        assert_close(value, REFERENCE[prefix + key], '{} / {}'.format(case['name'], key))
