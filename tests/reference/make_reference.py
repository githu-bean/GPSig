"""
Records reference outputs of the ORIGINAL TensorFlow 1.15 / GPflow 1.5 implementation of gpsig
(with the bug fixes of the port applied, see apply_bug_fixes).

Usage (in a Python 3.7 env with tensorflow==1.15.x, gpflow==1.5.1):
    python make_reference.py <dir containing the original gpsig package> <output .npz>

The ported code is checked against the resulting file by tests/test_regression.py.
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

src_dir, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.abspath(src_dir))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import gpflow
import gpsig

import cases as C


def apply_bug_fixes():
    """
    Applies the bug fixes of the port to the original code, so that the reference outputs remain directly comparable:
    - in SignatureKernel.K_seq_n_seq_covs (used with InducingSequences), with normalization and full_X2_cov=False,
      the rows of the cross-covariance were divided by the square roots of the diagonal of Kxx twice
    """
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(gpsig.kernels.SignatureKernel.K_seq_n_seq_covs))
    buggy = 'Kxx2_lvls /= Kxx_lvls_diag_sqrt[:, :, None] * Kx2x2_lvls_diag_sqrt[:, None, :]'
    assert src.count(buggy) == 1
    namespace = {}
    exec(src.replace(buggy, 'Kxx2_lvls /= Kx2x2_lvls_diag_sqrt[:, None, :]'), vars(gpsig.kernels), namespace)
    gpsig.kernels.SignatureKernel.K_seq_n_seq_covs = namespace['K_seq_n_seq_covs']


apply_bug_fixes()


def run_kernel_case(case):
    gpflow.reset_default_graph_and_session()
    d = C.kernel_data(case)
    kern = getattr(gpsig.kernels, case['kern'])(d['input_dim'], C.NUM_FEATURES, C.NUM_LEVELS, **case['kwargs'])
    for name, value in C.kernel_param_values(case).items():
        getattr(kern, name).assign(value)
    kern.compile()
    sess = kern.enquire_session()
    X, X2, Z, Z_incr, Zs = (tf.constant(d[k]) for k in ['X', 'X2', 'Z', 'Z_incr', 'Zs'])

    def base_kern_symm():
        # same computation as kern.compute_base_kern_symm, but inside a params_as_tensors context, since some
        # base kernels (SignatureMix._mix) lack @params_as_tensors in the original code and fail in the autoflow
        n = tf.shape(X)[0]
        Xr = tf.reshape(X, (n, -1, kern.num_features))
        len_examples = tf.shape(Xr)[1]
        Xr = tf.reshape(kern._apply_scaling_and_lags_to_sequences(Xr), (-1, kern.num_features))
        return tf.transpose(tf.reshape(kern._base_kern(Xr), [n, len_examples, n, len_examples]), [0, 2, 1, 3])

    autoflows = {
        'K_XX': lambda: kern.compute_K_symm(d['X']),
        'K_XX2': lambda: kern.compute_K(d['X'], d['X2']),
        'K_lvls_XX2': lambda: kern.compute_K_levels(d['X'], d['X2']),
        'Kdiag': lambda: kern.compute_Kdiag(d['X']),
        'Kdiag_lvls': lambda: kern.compute_K_level_diags(d['X']),
        'K_tens': lambda: kern.compute_K_tens(d['Z']),
        'K_incr_tens': lambda: kern.compute_K_incr_tens(d['Z_incr']),
        'K_tens_vs_seq': lambda: kern.compute_K_tens_vs_seq(d['Z'], d['X']),
        'K_incr_tens_vs_seq': lambda: kern.compute_K_incr_tens_vs_seq(d['Z_incr'], d['X']),
    }
    graph_fns = {
        'base_kern_XX': lambda: (base_kern_symm(),),
        'tens_covs': lambda: kern.K_tens_n_seq_covs(Z, X),
        'tens_covs_full': lambda: kern.K_tens_n_seq_covs(Z, X, full_X_cov=True),
        'incr_tens_covs': lambda: kern.K_tens_n_seq_covs(Z_incr, X, increments=True),
        'seq_covs': lambda: kern.K_seq_n_seq_covs(Zs, X),
    }

    # each output is computed separately, so that outputs on which the original code fails are recorded as such
    out, failed = {}, []
    for key, fn in autoflows.items():
        try:
            out[key] = fn()
        except Exception as e:
            failed.append(key)
            print('  [original fails] {}: {}: {}'.format(key, type(e).__name__, str(e).splitlines()[0][:150]))
    for prefix, fn in graph_fns.items():
        keys = [prefix] if prefix == 'base_kern_XX' else [prefix + s for s in ['_Kzz', '_Kzx', '_Kxx']]
        try:
            with gpflow.params_as_tensors_for(kern):
                tensors = fn()
            for key, value in zip(keys, sess.run(tensors)):
                out[key] = value
        except Exception as e:
            failed += keys
            print('  [original fails] {}: {}: {}'.format(prefix, type(e).__name__, str(e).splitlines()[0][:150]))
    assert sorted(list(out) + failed) == sorted(C.KERNEL_OUTPUTS)
    return out, failed


def run_model_case(case):
    gpflow.reset_default_graph_and_session()
    d = C.model_data(case)
    kern = gpsig.kernels.SignatureRBF(d['input_dim'], C.NUM_FEATURES, C.NUM_LEVELS, num_lags=case['num_lags'])
    for name, value in d['kern_values'].items():
        getattr(kern, name).assign(value)
    feat = getattr(gpsig.inducing_variables, case['feat'])(d['Z'], num_levels=C.NUM_LEVELS, **case['feat_kwargs'])
    if case['lik'] == 'Gaussian':
        lik = gpflow.likelihoods.Gaussian()
        lik.variance.assign(d['lik_variance'])
    elif case['lik'] == 'Bernoulli':
        lik = gpflow.likelihoods.Bernoulli()
    else:
        lik = gpflow.likelihoods.MultiClass(C.NUM_CLASSES)
    m = gpsig.models.SVGP(d['X'], d['Y'], kern=kern, feat=feat, likelihood=lik, num_latent=d['num_latent'], whiten=case['whiten'])
    m.q_mu.assign(d['q_mu'])
    m.q_sqrt.assign(d['q_sqrt'])
    if case['feat_kwargs'].get('learn_weights'):
        m.feature.W.assign(d['W'])

    out = {}
    out['elbo'] = m.compute_log_likelihood()
    out['f_mean'], out['f_var'] = m.predict_f(d['X_test'])
    out['y_mean'], out['y_var'] = m.predict_y(d['X_test'])
    out['log_density'] = m.predict_density(d['X_test'], d['Y_test'])

    params = {'variances': m.kern.variances, 'sigma': m.kern.sigma, 'lengthscales': m.kern.lengthscales,
              'Z': m.feature.Z, 'q_mu': m.q_mu, 'q_sqrt': m.q_sqrt}
    if case['num_lags']:
        params.update(lags=m.kern.lags, gamma=m.kern.gamma)
    if case['feat_kwargs'].get('learn_weights'):
        params['W'] = m.feature.W
    names = sorted(params)
    grads = tf.gradients(m.likelihood_tensor, [params[n].constrained_tensor for n in names])
    for n, g in zip(names, m.enquire_session().run(grads)):
        out['grad_' + n] = g
    return out


def main():
    results = {}
    print('gpsig reference: tf {}, gpflow {}, numpy {}'.format(tf.__version__, gpflow.__version__, np.__version__))
    for case in C.kernel_cases():
        out, failed = run_kernel_case(case)
        for key, value in out.items():
            results['kern/{}/{}'.format(case['name'], key)] = np.asarray(value)
        results['kern/{}/_failed'.format(case['name'])] = np.asarray(failed, dtype=str)
        print('kernel case done:', case['name'])
    for case in C.model_cases():
        for key, value in run_model_case(case).items():
            results['model/{}/{}'.format(case['name'], key)] = np.asarray(value)
        print('model case done:', case['name'])
    results['meta/versions'] = np.asarray('tf {} / gpflow {} / numpy {}'.format(tf.__version__, gpflow.__version__, np.__version__))
    results['meta/bug_fixes'] = np.asarray(apply_bug_fixes.__doc__)
    np.savez_compressed(out_path, **results)
    print('saved {} arrays to {}'.format(len(results), out_path))


if __name__ == '__main__':
    main()
