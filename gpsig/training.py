import time

import numpy as np
import tensorflow as tf
import gpflow

from gpflow.keras import tf_keras
from gpflow.utilities import parameter_dict

### Optimizers with the same names as the TF1 optimizers used with GPflow 1 (e.g. gpsig.training.NadamOptimizer(1e-3))

class _Optimizer:
    """
    Thin wrapper around a tf-keras optimizer, constructed with the same call signature as the TF1 optimizers.
    """
    _keras_class = None

    def __init__(self, learning_rate=0.001, **kwargs):
        self.learning_rate = learning_rate
        # XLA-compiled updates (the tf-keras default) intermittently produce NaNs with float64 variables on GPU
        kwargs.setdefault('jit_compile', False)
        self.optimizer = self._keras_class(learning_rate=learning_rate, **kwargs)

    def make_optimize_step(self, model, var_list):
        """
        Returns a function that takes a batch of data, performs one optimization step on var_list and returns the ELBO.
        """
        @tf.function
        def step(batch):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(var_list)
                loss = model.training_loss(batch)
            grads = tape.gradient(loss, var_list)
            self.optimizer.apply_gradients(zip(grads, var_list))
            return -loss
        return step

def _register_optimizer(name, keras_class):
    globals()[name] = type(name, (_Optimizer, ), {'_keras_class': keras_class})

for _name, _keras_name in [('AdamOptimizer', 'Adam'), ('NadamOptimizer', 'Nadam'), ('AdamaxOptimizer', 'Adamax'),
                           ('AdagradOptimizer', 'Adagrad'), ('AdadeltaOptimizer', 'Adadelta'), ('RMSPropOptimizer', 'RMSprop'),
                           ('GradientDescentOptimizer', 'SGD'), ('FtrlOptimizer', 'Ftrl'), ('AdamWOptimizer', 'AdamW')]:
    if hasattr(tf_keras.optimizers, _keras_name):
        _register_optimizer(_name, getattr(tf_keras.optimizers, _keras_name))


### Helpers for training models

def _trainable_variables(var_list):
    """
    Converts a list of gpflow Parameters / tf.Variables / Modules into a list of (unconstrained) tf.Variables.
    """
    variables = []
    for var in var_list:
        if isinstance(var, gpflow.Parameter):
            variables.append(var.unconstrained_variable)
        elif isinstance(var, tf.Module):
            variables += list(var.trainable_variables)
        else:
            variables.append(var)
    return variables

def _read_params(model, var_list=None):
    if var_list is None:
        return {path: param.numpy() for path, param in parameter_dict(model).items()}
    return [np.asarray(var.numpy()) for var in var_list]

def _save_snapshot(model, current_iter, current_time, elbo, history, val_scorer, save_params, callbacks, save_best_params, var_list, lower_is_better, patience):
    """
    Records the current state in history; returns True if training should be stopped early.
    """
    history[current_iter] = {}
    history[current_iter]['time'] = current_time
    history[current_iter]['elbo'] = elbo
    print('\t|\tELBO: {:.2f}'.format(elbo), end = '')
    if save_params:
        history[current_iter]['params'] = _read_params(model)

    if callbacks is not None:
        if isinstance(callbacks, list):
            history[current_iter]['saved'] = []
            for i, callback in enumerate(callbacks):
                history[current_iter]['saved'].append(callback(model))
        else:
            history[current_iter]['saved'] = [callbacks(model)]

    stop = False
    if val_scorer is not None:
        if isinstance(val_scorer, list):
            scores = []
            for i, scorer in enumerate(val_scorer):
                score = scorer(model)
                print('\t|\tVal. {}: {:.4f}'.format(i, score), end = '')
                scores.append(score)
            history[current_iter]['val'] = scores
            score = scores[-1]
        else:
            score = val_scorer(model)
            print('\t|\tVal. : {:.4f}'.format(score), end = '')
            history[current_iter]['val'] = score
            scores = score

        if save_best_params:
            if 'best' in history:
                if isinstance(history['best']['val'], list):
                    best_so_far = history['best']['val'][-1]
                else:
                    best_so_far = history['best']['val']

            save_current_params = False
            if 'best' not in history:
                history['best'] = {}
                save_current_params = True
            elif (lower_is_better and score <= best_so_far) or (not lower_is_better and score >= best_so_far):
                save_current_params = True

            if save_current_params:
                history['best']['iter'] = current_iter
                history['best']['time'] = current_time
                history['best']['elbo'] = elbo
                history['best']['val'] = scores
                history['best']['params'] = _read_params(model, var_list)

        if patience is not None:
            best_iter = history['best']['iter']
            if current_iter - best_iter > patience:
                print('\nNo improvement over validation loss has occured for {} iterations: stopping early...'.format(patience))
                stop = True

    print()
    return stop


def optimize(model, opt, max_iter=1000, print_freq=1, save_freq=50, val_scorer=None, history=None, callbacks=None,
                save_params=False, start_iter=0, global_step=None, var_list=None, save_best_params=False, lower_is_better=False, patience=None):
    """
    Optimizes the ELBO of model with opt (or a list of optimizers, one for each list of variables in var_list).
    The ELBO is evaluated on the next minibatch of the training data at every iteration.
    Every save_freq iterations, a snapshot is stored in history (ELBO, time, validation scores, ...), which is returned;
    passing the returned history to a subsequent call continues the optimization (iteration count, time, best params).
    The global_step argument is kept for compatibility and is ignored.
    """

    if isinstance(opt, list):
        assert isinstance(var_list, list)
        assert len(var_list) == len(opt) or len(var_list) + 1 == len(opt)
        var_groups = [_trainable_variables(vars) for vars in var_list]
        if len(var_list) + 1 == len(opt):
            considered_vars = set(id(var) for vars in var_groups for var in vars)
            var_groups.append([var for var in model.trainable_variables if id(var) not in considered_vars])
        steps = [o.make_optimize_step(model, vars) for o, vars in zip(opt, var_groups)]
        var_list = [var for vars in var_groups for var in vars]
    else:
        if var_list is None:
            steps = [opt.make_optimize_step(model, list(model.trainable_variables))]
        else:
            var_list = _trainable_variables(var_list)
            steps = [opt.make_optimize_step(model, var_list)]

    if history is None or len([x for x in history.keys() if str(x).isnumeric()]) == 0:
        history = {}
        start_iter = 0
        start_time = 0.0
    else:
        start_iter = max([x for x in history.keys() if str(x).isnumeric()])
        start_time = history[start_iter]['time']


    if 'best' in history:
        history['best']['iter'] = start_iter
        history['best']['time'] = start_time
        history['best']['params'] = _read_params(model, var_list)

    if start_iter == 0:
        print('-------------------------')
        print('  Starting optimization  ')
        print('-------------------------')
    else:
        print('---------------------------')
        print('  Continuing optimization  ')
        print('---------------------------')

    loop_start = time.time()
    for iteration in range(start_iter + 1, start_iter + max_iter + 1):
        for step in steps:
            step(model.next_batch())

        current_time = time.time() - loop_start + start_time
        if iteration % print_freq == 0 or iteration == max_iter + 1:
            print('\rIteration {}'.format(iteration), end='')
            print('\t|\tTime: {:.2f}'.format(current_time), end='')
        if iteration % save_freq == 0 or iteration == max_iter + 1:
            elbo = -model.training_loss(model.next_batch()).numpy()
            if _save_snapshot(model, iteration, current_time, elbo, history, val_scorer, save_params, callbacks,
                              save_best_params, var_list, lower_is_better, patience):
                break

    print('\nOptimization session finished...')

    return history
