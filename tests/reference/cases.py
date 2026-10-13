"""
Framework-agnostic definition of the regression test cases.

Imported both by `make_reference.py` (run with the original TF1/GPflow1 code to record reference outputs)
and by `tests/test_regression.py` (run with the ported TF2/GPflow2 code to compare against them),
so it must only depend on numpy and stay Python 3.7 compatible.
"""
import numpy as np

NUM_FEATURES = 2
NUM_LEVELS = 3
LEN_TENSORS = NUM_LEVELS * (NUM_LEVELS + 1) // 2

KERNEL_TYPES = ['SignatureLinear', 'SignatureCosine', 'SignaturePoly', 'SignatureRBF', 'SignatureMix',
                'SignatureMatern12', 'SignatureMatern32', 'SignatureMatern52', 'SignatureSpectral']


def kernel_cases():
    """
    Returns a list of dicts, each with keys 'name', 'kern' (class name in gpsig.kernels) and 'kwargs'.
    """
    cases = []
    for kern in KERNEL_TYPES:
        cases.append(dict(name='{}-default'.format(kern), kern=kern, kwargs={}))
    for kern in ['SignatureRBF', 'SignatureLinear']:
        for normalization in [True, False]:
            for difference in [True, False]:
                for order in [1, 2, NUM_LEVELS]:
                    for num_lags in [None, 2]:
                        name = '{}-norm{}-diff{}-order{}-lags{}'.format(kern, int(normalization), int(difference), order, num_lags or 0)
                        cases.append(dict(name=name, kern=kern, kwargs=dict(normalization=normalization, difference=difference,
                                                                           order=order, num_lags=num_lags)))
    return cases


def kernel_data(case, seed=0):
    """
    Inputs for a kernel case: sequences X, X2 (flattened to 2d as expected by the kernel), inducing tensors Z, Z_incr and inducing sequences Zs.
    """
    rng = np.random.RandomState(seed)
    num_lags = case['kwargs'].get('num_lags') or 0
    dim_tens = NUM_FEATURES * (num_lags + 1)
    len_examples, len_examples2 = 6, 6
    X = rng.randn(5, len_examples, NUM_FEATURES).cumsum(axis=1) / 2.
    X2 = rng.randn(4, len_examples2, NUM_FEATURES).cumsum(axis=1) / 2.
    Z = rng.randn(LEN_TENSORS, 3, dim_tens)
    Z_incr = rng.randn(LEN_TENSORS, 3, 2, dim_tens)
    Zs = rng.randn(3, 4, NUM_FEATURES).cumsum(axis=1) / 2.
    return dict(X=X.reshape([X.shape[0], -1]), X2=X2.reshape([X2.shape[0], -1]), Z=Z, Z_incr=Z_incr,
                Zs=Zs.reshape([Zs.shape[0], -1]), input_dim=len_examples * NUM_FEATURES)


def kernel_param_values(case, seed=1):
    """
    Non-default (constrained) values for all kernel hyperparameters, keyed by attribute name.
    """
    rng = np.random.RandomState(seed)
    num_lags = case['kwargs'].get('num_lags') or 0
    values = dict(variances=np.exp(0.3 * rng.randn(NUM_LEVELS + 1)), sigma=np.array(1.3))
    if case['kern'] == 'SignatureSpectral':
        Q = 5
        values.update(alpha=np.exp(0.3 * rng.randn(Q)), omega=np.exp(0.3 * rng.randn(Q, NUM_FEATURES)),
                      gamma=np.exp(0.3 * rng.randn(Q, NUM_FEATURES)))
    else:
        values['lengthscales'] = np.exp(0.3 * rng.randn(NUM_FEATURES))
    if num_lags > 0:
        values['lags'] = np.sort(rng.uniform(0.05, 0.4, size=num_lags))
        values['gamma'] = np.exp(0.3 * rng.randn(num_lags + 1))
    if case['kern'] == 'SignaturePoly':
        values['gamma'] = np.array(0.7)
    if case['kern'] == 'SignatureMix':
        values['mixing'] = np.array(0.35)
    return values


# names of kernel outputs recorded for every kernel case
KERNEL_OUTPUTS = ['K_XX', 'K_XX2', 'K_lvls_XX2', 'Kdiag', 'Kdiag_lvls', 'base_kern_XX',
                  'K_tens', 'K_incr_tens', 'K_tens_vs_seq', 'K_incr_tens_vs_seq',
                  'tens_covs_Kzz', 'tens_covs_Kzx', 'tens_covs_Kxx',
                  'tens_covs_full_Kzz', 'tens_covs_full_Kzx', 'tens_covs_full_Kxx',
                  'incr_tens_covs_Kzz', 'incr_tens_covs_Kzx', 'incr_tens_covs_Kxx',
                  'seq_covs_Kzz', 'seq_covs_Kzx', 'seq_covs_Kxx']


def model_cases():
    """
    SVGP model cases: likelihood x inducing variable type x whitening.
    """
    cases = []
    for lik in ['Gaussian', 'Bernoulli', 'MultiClass']:
        for feat, feat_kwargs in [('InducingTensors', dict(increments=False)), ('InducingTensors', dict(increments=True)),
                                  ('InducingTensors', dict(increments=True, learn_weights=True)),
                                  ('InducingSequences', dict()), ('InducingSequences', dict(learn_weights=True))]:
            for whiten in [True, False]:
                tag = ''.join('-{}{}'.format(k, int(v)) for k, v in sorted(feat_kwargs.items()))
                name = '{}-{}{}-whiten{}'.format(lik, feat, tag, int(whiten))
                cases.append(dict(name=name, lik=lik, feat=feat, feat_kwargs=feat_kwargs, whiten=whiten, num_lags=None))
    # one case exercising the lags parameters inside the model and its gradients
    cases.append(dict(name='Gaussian-InducingTensors-increments1-lags1-whiten1', lik='Gaussian', feat='InducingTensors',
                      feat_kwargs=dict(increments=True), whiten=True, num_lags=1))
    return cases


NUM_CLASSES = 3


def model_data(case, seed=2):
    rng = np.random.RandomState(seed)
    num_lags = case['num_lags'] or 0
    len_examples, num_inducing = 5, 4
    X = rng.randn(12, len_examples, NUM_FEATURES).cumsum(axis=1) / 2.
    X_test = rng.randn(7, len_examples, NUM_FEATURES).cumsum(axis=1) / 2.
    if case['lik'] == 'Gaussian':
        Y = rng.randn(12, 1)
        Y_test = rng.randn(7, 1)
        num_latent = 1
    elif case['lik'] == 'Bernoulli':
        Y = rng.randint(0, 2, size=(12, 1)).astype(np.float64)
        Y_test = rng.randint(0, 2, size=(7, 1)).astype(np.float64)
        num_latent = 1
    else:
        Y = rng.randint(0, NUM_CLASSES, size=(12, 1)).astype(np.float64)
        Y_test = rng.randint(0, NUM_CLASSES, size=(7, 1)).astype(np.float64)
        num_latent = NUM_CLASSES
    dim_tens = NUM_FEATURES * (num_lags + 1)
    if case['feat'] == 'InducingTensors':
        if case['feat_kwargs'].get('increments'):
            Z = rng.randn(LEN_TENSORS, num_inducing, 2, dim_tens)
        else:
            Z = rng.randn(LEN_TENSORS, num_inducing, dim_tens)
    else:
        Z = rng.randn(num_inducing, 3, NUM_FEATURES).cumsum(axis=1) / 2.
    q_mu = 0.5 * rng.randn(num_inducing, num_latent)
    q_sqrt = np.tril(0.3 * rng.randn(num_latent, num_inducing, num_inducing)) + 0.5 * np.eye(num_inducing)[None]
    W = np.eye(num_inducing)[None] + 0.2 * rng.randn(NUM_LEVELS, num_inducing, num_inducing)
    kern_values = dict(variances=np.exp(0.3 * rng.randn(NUM_LEVELS + 1)), sigma=np.array(1.3),
                       lengthscales=np.exp(0.3 * rng.randn(NUM_FEATURES)))
    if num_lags > 0:
        kern_values['lags'] = np.sort(rng.uniform(0.05, 0.4, size=num_lags))
        kern_values['gamma'] = np.exp(0.3 * rng.randn(num_lags + 1))
    return dict(X=X.reshape([12, -1]), Y=Y, X_test=X_test.reshape([7, -1]), Y_test=Y_test, Z=Z, q_mu=q_mu, q_sqrt=q_sqrt, W=W,
                kern_values=kern_values, lik_variance=np.array(0.7), num_latent=num_latent, input_dim=len_examples * NUM_FEATURES)


# names of model outputs; grads are d(ELBO)/d(constrained parameter value)
MODEL_OUTPUTS = ['elbo', 'f_mean', 'f_var', 'y_mean', 'y_var', 'log_density']
MODEL_GRADS = ['variances', 'sigma', 'lengthscales', 'lags', 'gamma', 'Z', 'W', 'q_mu', 'q_sqrt']
