import numpy as np
import tensorflow as tf

from gpflow import models
from gpflow.config import default_float, default_jitter
from gpflow.kullback_leiblers import gauss_kl
from gpflow.conditionals.util import base_conditional, expand_independent_outputs
from gpflow.utilities import multiple_assign

from .inducing_variables import InducingTensors, InducingSequences, Kuu_Kuf_Kff, Kuu, Kuf

class SVGP(models.SVGP):
    """
    Re-implementation of SVGP from GPflow with a few minor tweaks. Slightly more efficient with SignatureKernels, and when using the low-rank option with signature kernels, this code must be used.

    As in GPflow 1, the training data is stored in the model (X, Y), and if minibatch_size is given, the ELBO is evaluated on minibatches of it.
    """
    def __init__(self, X, Y, kern, likelihood, feat, mean_function=None, num_latent=None, q_diag=False, whiten=True, minibatch_size=None, num_data=None, q_mu=None, q_sqrt=None, shuffle=True, **kwargs):

        if not isinstance(feat, InducingTensors) and not isinstance(feat, InducingSequences):
            raise ValueError('feat must be of type either InducingTensors or InducingSequences')

        num_latent = num_latent or (Y.shape[1] if Y.ndim > 1 else 1)
        super().__init__(kern, likelihood, feat, mean_function=mean_function, num_latent_gps=num_latent, q_diag=q_diag,
                         q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten, num_data=num_data or X.shape[0], **kwargs)
        self.minibatch_size = minibatch_size
        self.shuffle = shuffle
        self.X, self.Y = X, Y
        self._batches = None

        return

    ### GPflow 1 style attribute names and data handling

    @property
    def kern(self):
        return self.kernel

    @property
    def feature(self):
        return self.inducing_variable

    @property
    def num_latent(self):
        return self.num_latent_gps

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        self._X = X
        self._batches = None

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, Y):
        # GPflow 1 also accepted labels of shape (num_data,), GPflow 2 requires (num_data, num_outputs)
        self._Y = Y[:, None] if Y.ndim == 1 else Y
        self._batches = None

    def next_batch(self):
        """
        Returns the data the ELBO is evaluated on during training: all data if minibatch_size is None, the next minibatch otherwise.
        """
        if self.minibatch_size is None:
            return tf.constant(self._X, dtype=default_float()), tf.constant(self._Y, dtype=default_float())
        if self._batches is None:
            if self._X.shape[0] != self._Y.shape[0]:
                raise ValueError('X and Y must contain the same number of examples.')
            dataset = tf.data.Dataset.from_tensor_slices((np.asarray(self._X, dtype=default_float()), np.asarray(self._Y, dtype=default_float())))
            if self.shuffle:
                dataset = dataset.shuffle(self._X.shape[0], seed=0, reshuffle_each_iteration=True)
            self._batches = iter(dataset.repeat().batch(self.minibatch_size))
        return next(self._batches)

    def assign(self, values):
        """
        Assigns parameter values from a dict {path: value}, e.g. as stored in history['best']['params'] by gpsig.training.optimize.
        """
        multiple_assign(self, values)

    ### Objective and predictions

    def elbo(self, data=None):
        """
        Evidence lower bound on data=(X, Y); if data is None, it is evaluated on the next batch of the stored training data.
        """
        X, Y = self.next_batch() if data is None else data

        num_samples = tf.shape(X)[0]

        if self.whiten:
            f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
            KL =  gauss_kl(self.q_mu, tf.linalg.band_part(self.q_sqrt, -1, 0))
        else:
            f_mean, f_var, Kzz = self.predict_f(X, full_cov=False, full_output_cov=False, return_Kzz=True)
            KL =  gauss_kl(self.q_mu, tf.linalg.band_part(self.q_sqrt, -1, 0), K=Kzz)

        # compute variational expectations
        var_exp = self.likelihood.variational_expectations(X, f_mean, f_var, Y)

        # scaling for batch size
        scale = tf.cast(self.num_data, default_float()) / tf.cast(num_samples, default_float())
        return tf.reduce_sum(var_exp) * scale - KL

    def maximum_log_likelihood_objective(self, data=None):
        return self.elbo(data)

    def predict_f(self, X_new, full_cov=False, full_output_cov=False, return_Kzz=False):

        X_new = tf.convert_to_tensor(X_new, dtype=default_float())
        Kzz, Kzx, Kxx = Kuu_Kuf_Kff(self.inducing_variable, self.kernel, X_new, jitter=default_jitter(), full_f_cov=full_cov)
        f_mean, f_var = base_conditional(Kzx, Kzz, Kxx, self.q_mu, full_cov=full_cov, q_sqrt=tf.linalg.band_part(self.q_sqrt, -1, 0), white=self.whiten)
        f_mean += self.mean_function(X_new)
        f_var = expand_independent_outputs(f_var, full_cov, full_output_cov)

        if return_Kzz:
            return f_mean, f_var, Kzz
        else:
            return f_mean, f_var

    ### GPflow 1 style methods returning numpy arrays

    def compute_log_likelihood(self):
        """
        ELBO on all of the stored training data.
        """
        return self.elbo((tf.constant(self._X, dtype=default_float()), tf.constant(self._Y, dtype=default_float()))).numpy()

    def predict_density(self, X_new, Y_new):
        """
        Log predictive density of each (X_new, Y_new) pair.
        """
        X_new = tf.convert_to_tensor(X_new, dtype=default_float())
        Y_new = tf.convert_to_tensor(Y_new, dtype=default_float())
        f_mean, f_var = self.predict_f(X_new, full_cov=False, full_output_cov=False)
        return self.likelihood.predict_log_density(X_new, f_mean, f_var, Y_new).numpy()
