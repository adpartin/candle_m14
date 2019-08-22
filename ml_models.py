""" This script contains various ML models and some utility functions. """
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time
from collections import OrderedDict
import math

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Embedding, Flatten, Lambda, merge
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard

try:
    import lightgbm as lgb
except ImportError:
    print('Module not found (lightgbm).')


def clr_keras_callback(mode=None, base_lr=1e-4, max_lr=1e-3, gamma=0.999994):
    """ Creates keras callback for cyclical learning rate. """
    # keras_contrib = './keras_contrib/callbacks'
    # sys.path.append(keras_contrib)
    from cyclical_learning_rate import CyclicLR

    if mode == 'trng1':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular')
    elif mode == 'trng2':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular2')
    elif mode == 'exp':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='exp_range', gamma=gamma) # 0.99994; 0.99999994; 0.999994
    return clr


def r2_krs(y_true, y_pred):
    # from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def get_model(model_name, init_kwargs=None):
    """ Return a model.
    Args:
        init_kwargs : init parameters to the model
        model_name : model name
    """
    if model_name == 'lgb_reg':
        model = LGBM_REGRESSOR(**init_kwargs)
    elif model_name == 'rf_reg':
        model = RF_REGRESSOR(**init_kwargs)
    elif model_name == 'nn_reg':
        model = KERAS_REGRESSOR(**init_kwargs)
    elif model_name == 'nn_reg0':
        model = NN_REG0(**init_kwargs)
    elif model_name == 'nn_reg1':
        model = NN_REG1(**init_kwargs)
    elif model_name == 'nn_reg2':
        model = NN_REG2(**init_kwargs)
    elif model_name == 'nn_reg3':
        model = NN_REG3(**init_kwargs)
    elif model_name == 'nn_reg4':
        model = NN_REG4(**init_kwargs)
    else:
        raise ValueError('model_name is invalid.')
    return model


def save_krs_history(history, outdir='.'):
    fname = 'krs_history.csv'
    hh = pd.DataFrame(history.history)
    hh['epoch'] = np.asarray(history.epoch) + 1    
    hh.to_csv( Path(outdir)/fname, index=False )
    return hh


def capitalize_metric(met):
    return ' '.join(s.capitalize() for s in met.split('_'))


def plot_prfrm_metrics(history, title=None, skp_ep=0, outdir='.', add_lr=False):
    """ Plots keras training curves history.
    Args:
        skp_ep: number of epochs to skip when plotting metrics 
        add_lr: add curve of learning rate progression over epochs
    """
    all_metrics = list(history.history.keys())
    pr_metrics = ['_'.join(m.split('_')[1:]) for m in all_metrics if 'val' in m]

    epochs = np.asarray(history.epoch) + 1
    if len(epochs) <= skp_ep: skp_ep = 0
    eps = epochs[skp_ep:]
    hh = history.history
        
    for p, m in enumerate(pr_metrics):
        metric_name = m
        metric_name_val = 'val_' + m

        y_tr = hh[metric_name][skp_ep:]
        y_vl = hh[metric_name_val][skp_ep:]
        
        ymin = min(set(y_tr).union(y_vl))
        ymax = max(set(y_tr).union(y_vl))
        lim = (ymax - ymin) * 0.1
        ymin, ymax = ymin - lim, ymax + lim

        # Start figure
        fig, ax1 = plt.subplots()
        
        # Plot metrics
        # ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=metric_name)
        # ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=1, alpha=0.6, label=metric_name_val)
        ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name))
        ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name_val))
        ax1.set_xlabel('Epoch')
        # ylabel = ' '.join(s.capitalize() for s in metric_name.split('_'))
        ax1.set_ylabel(capitalize_metric(metric_name))
        ax1.set_xlim([min(eps)-1, max(eps)+1])
        ax1.set_ylim([ymin, ymax])
        ax1.tick_params('y', colors='k')
        
        # ax1.tick_params(axis='both', which='major', labelsize=12)
        # ax1.tick_params(axis='both', which='minor', labelsize=12)        
        
        # Add learning rate
        if (add_lr is True) and ('lr' in hh):            
            ax2 = ax1.twinx()
            ax2.plot(eps, hh['lr'][skp_ep:], color='g', marker='.', linestyle=':', linewidth=1,
                     alpha=0.6, markersize=5, label='LR')
            ax2.set_ylabel('Learning rate', color='g', fontsize=12)
            
            ax2.set_yscale('log') # 'linear'
            ax2.tick_params('y', colors='g')
        
        ax1.grid(True)
        # plt.legend([metric_name, metric_name_val], loc='best')
        # medium.com/@samchaaa/how-to-plot-two-different-scales-on-one-plot-in-matplotlib-with-legend-46554ba5915a
        legend = ax1.legend(loc='best', prop={'size': 10})
        frame = legend.get_frame()
        frame.set_facecolor('0.95')
        if title is not None: plt.title(title)
        
        # fig.tight_layout()
        figpath = Path(outdir) / (metric_name+'.png')
        plt.savefig(figpath, bbox_inches='tight')
        plt.close()
        

def plot_metrics_from_logs(path_to_logs, title=None, name=None, skp_ep=0, outdir='.'):
    """ Plots keras training from logs.
    Args:
        path_to_logs : full path to log file
        skp_ep: number of epochs to skip when plotting metrics 
    """
    history = pd.read_csv(path_to_logs, sep=',', header=0)
    
    all_metrics = list(history.columns)
    pr_metrics = ['_'.join(m.split('_')[1:]) for m in all_metrics if 'val' in m]

    epochs = history['epoch'] + 1
    if len(epochs) <= skp_ep: skp_ep = 0
    eps = epochs[skp_ep:]
    hh = history
    
    for p, m in enumerate(pr_metrics):
        metric_name = m
        metric_name_val = 'val_' + m

        y_tr = hh[metric_name][skp_ep:]
        y_vl = hh[metric_name_val][skp_ep:]
        
        ymin = min(set(y_tr).union(y_vl))
        ymax = max(set(y_tr).union(y_vl))
        lim = (ymax - ymin) * 0.1
        ymin, ymax = ymin - lim, ymax + lim

        # Start figure
        fig, ax1 = plt.subplots()
        
        # Plot metrics
        # ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=metric_name)
        # ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=1, alpha=0.6, label=metric_name_val)
        ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name))
        ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name_val))        
        ax1.set_xlabel('Epoch')
        # ylabel = ' '.join(s.capitalize() for s in metric_name.split('_'))
        ax1.set_ylabel(capitalize_metric(metric_name))
        ax1.set_ylabel(ylabel)
        ax1.set_xlim([min(eps)-1, max(eps)+1])
        ax1.set_ylim([ymin, ymax])
        ax1.tick_params('y', colors='k')
        
        ax1.grid(True)
        # plt.legend([metric_name, metric_name_val], loc='best')
        # medium.com/@samchaaa/how-to-plot-two-different-scales-on-one-plot-in-matplotlib-with-legend-46554ba5915a
        legend = ax1.legend(loc='best', prop={'size': 10})
        frame = legend.get_frame()
        frame.set_facecolor('0.95')
        if title is not None: plt.title(title)
        
        # fig.tight_layout()
        if name is not None:
            fname = name + '_' + metric_name + '.png'
        else:
            fname = metric_name + '.png'
        figpath = Path(outdir) / fname
        plt.savefig(figpath, bbox_inches='tight')
        plt.close()
        
    return history
        
        
class Attention(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Attention, self).build(input_shape)
    
    def call(self, V):
        Q = keras.backend.dot(V, self.kernel)
        Q =  Q * V
        Q = Q / math.sqrt(self.output_dim)
        Q = keras.activations.softmax(Q)
        return Q
    
    def compute_output_shape(self, input_shape):
        return input_shape


class BaseMLModel():
    """ A parent class with some general methods for children ML classes.
    The children classes are specific ML models such random forest regressor, lightgbm regressor, etc.
    """
    def __adj_r2_score(self, ydata, preds):
        """ Calc adjusted r^2.
        https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
        https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/
        https://stats.stackexchange.com/questions/334004/can-r2-be-greater-than-1
        """
        r2_score = sklearn.metrics.r2_score(ydata, preds)
        adj_r2 = 1 - (1 - r2_score) * (self.x_size[0] - 1)/(self.x_size[0] - self.x_size[1] - 1)
        return adj_r2


    def build_dense_block(self, layers, inputs, name=''):
        """ This function only applicable to keras NNs. """
        for i, l_size in enumerate(layers):
            if i == 0:
                x = Dense(l_size, kernel_initializer=self.initializer, name=f'{name}.fc{i+1}.{l_size}')(inputs)
            else:
                x = Dense(l_size, kernel_initializer=self.initializer, name=f'{name}.fc{i+1}.{l_size}')(x)
            x = BatchNormalization(name=f'{name}.bn{i+1}')(x)
            x = Activation('relu', name=f'{name}.a{i+1}')(x)
            x = Dropout(self.dr_rate, name=f'{name}.drp{i+1}.{self.dr_rate}')(x)        
        return x

    

class NN_REG0(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg0'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        layers = [1000, 1000, 500, 250, 125, 60, 30]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model
        
        

class NN_REG1(BaseMLModel):
    """ Neural network regressor. 
    Fully-connected NN with attention layer.
    """
    model_name = 'nn_reg1'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', logger=None):
        inputs = Input(shape=(input_dim,))
        #x = Lambda(lambda x: x, output_shape=(1000,))(inputs)
        # attn_lin = Dense(1000, activation='relu', name='attn_lin')(inputs)
        # attn_probs = Dense(1000, activation='softmax', name='attn_probs')(inputs)
        # x = keras.layers.multiply( [attn_lin, attn_probs], name='attn')
        
        # New attention layer (Rick, Austin)
        a = Dense(1000)(inputs)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        b = Attention(1000)(a)
        x = keras.layers.multiply([b, a])

        x = Dense(1000)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(500)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(250)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(125)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(60)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(30)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        outputs = Dense(1, activation='relu')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model

        

class NN_REG3(BaseMLModel):
    """ Neural network regressor.
    Uno-style network.
    """
    model_name = 'nn_reg3'

    def __init__(self, in_dim_rna, in_dim_dsc, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        # https://keras.io/getting-started/functional-api-guide/
        # Chollet book
        
        #def create_tower(layers, inputs, name):

        #    for i, l_size in enumerate(layers):
        #        if i == 0:
        #            x = Dense(l_size, kernel_initializer=initializer, name=f'{name}_FC{i+1}')(inputs)
        #        else:
        #            x = Dense(l_size, kernel_initializer=initializer, name=f'{name}_FC{i+1}')(x)
        #        x = BatchNormalization(name=f'{name}_BN{i+1}')(x)
        #        x = Activation('relu', name=f'{name}_A{i+1}')(x)
        #        x = Dropout(dr_rate, name=f'{name}_DRP{i+1}')(x)        

        #    x = Model(inputs=inputs, outputs=x, name=f'out_{name}')
        #    return x

        in_rna = Input(shape=(in_dim_rna,), name='in_rna')
        out_rna = self.build_dense_block(layers=[1000,800,600], inputs=in_rna, name='rna')
        rna = Model(inputs=in_rna, outputs=out_rna, name=f'out_rna')
        
        in_dsc = Input(shape=(in_dim_dsc,), name='in_dsc')
        out_dsc = self.build_dense_block(layers=[1000,800,600], inputs=in_dsc, name='dsc')
        dsc = Model(inputs=in_dsc, outputs=out_dsc, name=f'out_dsc')

        # merged = merge.concatenate([rna.output, dsc.output])
        # x = create_tower(layers=[1000,800,600], input_dim=rna.output_shape[-1] + dsc.output_shape[-1], name='merged')

        """
        # RNA
        in_rna = Input(shape=(in_dim_rna,), name='in_rna')
        layers = [1000, 800, 600]

        for i, l_size in enumerate(layers):
            if i == 0:
                x = Dense(l_size, kernel_initializer=initializer, name=f'FC{i+1}')(in_rna)
            else:
                x = Dense(l_size, kernel_initializer=initializer, name=f'FC{i+1}')(x)
            x = BatchNormalization(name=f'BN{i+1}')(x)
            x = Activation('relu', name=f'A{i+1}')(x)
            x = Dropout(dr_rate, name=f'DRP{i+1}')(x)        

        rna = Model(inputs=in_rna, outputs=x, name='out_rna')
        del x

        # DSC
        in_dsc = Input(shape=(in_dim_dsc,), name='in_dsc')
        layers = [1000, 800, 600]

        for i, l_size in enumerate(layers):
            if i == 0:
                x = Dense(l_size, kernel_initializer=initializer, name=f'FC{i+1}')(in_dsc)
            else:
                x = Dense(l_size, kernel_initializer=initializer, name=f'FC{i+1}')(x)
            x = BatchNormalization(name=f'BN{i+1}')(x)
            x = Activation('relu', name=f'A{i+1}')(x)
            x = Dropout(dr_rate, name=f'DRP{i+1}')(x)        

        dsc = Model(inputs=in_dsc, outputs=x, name='out_dsc')
        del x
        """

        """
        # Proc rna
        in_rna = Input(shape=(in_dim_rna,), name='in_rna')
        a = Dense(1000)(in_rna)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        a = Dropout(dr_rate)(a)

        a = Dense(800)(a)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        a = Dropout(dr_rate)(a)
        
        a = Dense(600)(a)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        a = Dropout(dr_rate)(a)

        rna = Model(inputs=in_rna, outputs=a, name='out_rna')

        # Proc dsc
        in_dsc = Input(shape=(in_dim_dsc,), name='in_dsc')
        b = Dense(1000)(in_dsc)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)
        b = Dropout(dr_rate)(b)

        b = Dense(800)(b)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)
        b = Dropout(dr_rate)(b)

        b = Dense(600)(b)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)
        b = Dropout(dr_rate)(b)
        
        dsc = Model(inputs=in_dsc, outputs=b, name='out_dsc')
        """
        
        # Merge layers
        merged = merge.concatenate([rna.output, dsc.output])
        
        # Dense layers
        x = Dense(1000, name='in_merged')(merged)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(800)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(600)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        outputs = Dense(1, activation='relu', name='output')(x)
        model = Model(inputs=[in_rna, in_dsc], outputs=[outputs])
        
        if opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model


    def fit_cv(self,
            in_rna, in_dsc,
            epochs: int=150, batch_size: int=32,
            cv: int=5,
            cv_splits: tuple=None):
        # TODO: finish this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pass
        """
        for 
        model.fit({'in_rna': in_rna, 'in_dsc': in_dsc},
                  {'out_nn': ydata},
                    epochs=epochs, batch_size=batch_size)   

        if cv_splits is not None:
            tr_id = cv_splits[0]
            vl_id = cv_splits[1]
            assert tr_id.shape[1]==vl_id.shape[1], 'tr and vl must have the same of folds.'
            cv_folds = tr_id.shape[1]

            for i in range(tr_id.shape[1]):
                tr_dct[i] = tr_id.iloc[:, i].dropna().values.astype(int).tolist()
                vl_dct[i] = vl_id.iloc[:, i].dropna().values.astype(int).tolist()

            if tr_id.shape[1] == 1:
                vl_size = vl_id.shape[0]/(vl_id.shape[0] + tr_id.shape[0])

        # If pre-defined splits are not passed, then generate splits on the fly
        else:
            if isinstance(cv, int):
                cv_folds = cv
                cv = KFold(n_splits=cv_folds, shuffle=False, random_state=random_state)
            else:
                cv_folds = cv.get_n_splits() # cv is a sklearn splitter

            if cv_folds == 1:
                vl_size = cv.test_size
        """
        

class LGBM_REGRESSOR(BaseMLModel):
    """ LightGBM regressor. """
    ml_objective = 'regression'
    model_name = 'lgb_reg'

    def __init__(self, n_estimators=100, eval_metric=['l2', 'l1'], n_jobs=1, random_state=None, logger=None):
        # TODO: use config file to set default parameters (like in candle)
        
        self.model = lgb.LGBMModel(
            objective = LGBM_REGRESSOR.ml_objective,
            n_estimators = n_estimators,
            n_jobs = n_jobs,
            random_state = random_state)


    # def fit(self, X, y, eval_set=None, **fit_params):
    #     #self.eval_set = eval_set
    #     #self.X = X
    #     #self.y = y
    #     #self.x_size = X.shape  # this is used to calc adjusteed r^2
        
    #     t0 = time.time()
    #     self.model.fit(X, y,
    #                    eval_metric=self.eval_metric,
    #                    eval_set=eval_set,
    #                    **fit_params)
    #     self.train_runtime = time.time() - t0

    #     if self.logger is not None:
    #         self.logger.info('Train time: {:.2f} mins'.format(self.train_runtime/60))


    def dump_model(self, outdir='.'):
        # lgb_reg.save_model(os.path.join(run_outdir, 'lgb_'+ml_type+'_model.txt'))
        joblib.dump(self.model, filename=Path(outdir)/('model.' + LGBM_REGRESSOR.model_name + '.pkl'))
        # lgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))

        
    def plot_fi(self, max_num_features=20, title='LGBMRegressor', outdir=None):
        lgb.plot_importance(booster=self.model, max_num_features=max_num_features, grid=True, title=title)
        plt.tight_layout()

        filename = LGBM_REGRESSOR.model_name + '_fi.png'
        if outdir is None:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.savefig(Path(outdir)/filename, bbox_inches='tight')


    # # Plot training curves
    # # TODO: note, plot_metric didn't accept 'mae' although it's alias for 'l1' 
    # # TODO: plot_metric requires dict from train(), but train returns 'lightgbm.basic.Booster'??
    # for m in eval_metric:
    #     ax = lgb.plot_metric(booster=lgb_reg, metric=m, grid=True)
    #     plt.savefig(os.path.join(run_outdir, model_name+'_learning_curve_'+m+'.png'))
    
