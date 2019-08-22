import os
import sys
import platform
from pathlib import Path
from pprint import pprint, pformat

import sklearn
import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

SEED = 0


# File path
filepath = Path(__file__).resolve().parent


# Utils
from classlogger import Logger
from cv_splitter import cv_splitter, plot_ytr_yvl_dist


def split_size(x):
    """ Split size can be float (0, 1) or int (casts value as needed). """
    assert x > 0, 'Split size must be greater than 0.'
    return int(x) if x > 1.0 else x


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open( Path(outpath), 'w' ) as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))


def make_split(xdata, meta, outdir, args):
    # Data splits
    te_method = args['te_method']
    cv_method = args['cv_method']
    te_size = split_size(args['te_size'])
    vl_size = split_size(args['vl_size'])

    # Features 
    cell_fea = args['cell_fea']
    drug_fea = args['drug_fea']
    # fea_list = cell_fea + drug_fea
    
    # Other params
    n_jobs = args['n_jobs']

    # Hard split
    grp_by_col = None
    # cv_method = 'simple'

    # TODO: this need to be improved
    mltype = 'reg'  # required for the splits (stratify in case of classification)
    
    
    # -----------------------------------------------
    #       Outdir and Logger
    # -----------------------------------------------
    # Logger
    lg = Logger(outdir/'splitter.log')
    lg.logger.info(f'File path: {filepath}')
    lg.logger.info(f'\n{pformat(args)}')

    # Dump args to file
    dump_dict(args, outpath=outdir/'args.txt')


    # -----------------------------------------------
    #       Load data and pre-proc
    # -----------------------------------------------
    if (outdir/'xdata.parquet').is_file():
        xdata = pd.read_parquet( outdir/'xdata.parquet' )
        meta = pd.read_parquet( outdir/'meta.parquet' )


    # -----------------------------------------------
    #       Train-test split
    # -----------------------------------------------
    np.random.seed(SEED)
    idx_vec = np.random.permutation(xdata.shape[0])

    if te_method is not None:
        lg.logger.info('\nSplit train/test.')
        te_splitter = cv_splitter(cv_method=te_method, cv_folds=1, test_size=te_size,
                                  mltype=mltype, shuffle=False, random_state=SEED)

        te_grp = meta[grp_by_col].values[idx_vec] if te_method=='group' else None
        if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)
   
        # Split train/test
        tr_id, te_id = next(te_splitter.split(idx_vec, groups=te_grp))
        tr_id = idx_vec[tr_id] # adjust the indices!
        te_id = idx_vec[te_id] # adjust the indices!

        pd.Series(tr_id).to_csv(outdir/f'tr_id.csv', index=False, header=[0])
        pd.Series(te_id).to_csv(outdir/f'te_id.csv', index=False, header=[0])
        
        lg.logger.info('Train: {:.1f}'.format( len(tr_id)/xdata.shape[0] ))
        lg.logger.info('Test:  {:.1f}'.format( len(te_id)/xdata.shape[0] ))
        
        # Update the master idx vector for the CV splits
        idx_vec = tr_id

        # Plot dist of responses (TODO: this can be done to all response metrics)
        # plot_ytr_yvl_dist(ytr=tr_ydata.values, yvl=te_ydata.values,
        #         title='tr and te', outpath=run_outdir/'tr_te_resp_dist.png')

        # Confirm that group splits are correct
        if te_method=='group' and grp_by_col is not None:
            tr_grp_unq = set(meta.loc[tr_id, grp_by_col])
            te_grp_unq = set(meta.loc[te_id, grp_by_col])
            lg.logger.info(f'\tTotal group ({grp_by_col}) intersections btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}.')
            lg.logger.info(f'\tA few intersections : {list(tr_grp_unq.intersection(te_grp_unq))[:3]}.')

        # Update vl_size to effective vl_size
        vl_size = vl_size * xdata.shape[0]/len(tr_id)

        del tr_id, te_id


    # -----------------------------------------------
    #       Generate CV splits
    # -----------------------------------------------
    cv_folds_list = [1, 5, 7, 10, 15, 20]
    lg.logger.info(f'\nStart CV splits ...')
    
    for cv_folds in cv_folds_list:
        lg.logger.info(f'\nCV folds: {cv_folds}')

        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=vl_size,
                         mltype=mltype, shuffle=False, random_state=SEED)

        cv_grp = meta[grp_by_col].values[idx_vec] if cv_method=='group' else None
        if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
    
        tr_folds = {}
        vl_folds = {}

        # Start CV iters
        for fold, (tr_id, vl_id) in enumerate(cv.split(idx_vec, groups=cv_grp)):
            tr_id = idx_vec[tr_id] # adjust the indices!
            vl_id = idx_vec[vl_id] # adjust the indices!

            tr_folds[fold] = tr_id.tolist()
            vl_folds[fold] = vl_id.tolist()

            # Confirm that group splits are correct
            if cv_method=='group' and grp_by_col is not None:
                tr_grp_unq = set(meta.loc[tr_id, grp_by_col])
                vl_grp_unq = set(meta.loc[vl_id, grp_by_col])
                lg.logger.info(f'\tTotal group ({grp_by_col}) intersections btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}.')
                lg.logger.info(f'\tUnique cell lines in tr: {len(tr_grp_unq)}.')
                lg.logger.info(f'\tUnique cell lines in vl: {len(vl_grp_unq)}.')
        
        # Convet to df
        # from_dict takes too long  -->  faster described here: stackoverflow.com/questions/19736080/
        # tr_folds = pd.DataFrame.from_dict(tr_folds, orient='index').T 
        # vl_folds = pd.DataFrame.from_dict(vl_folds, orient='index').T
        tr_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in tr_folds.items() ]))
        vl_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in vl_folds.items() ]))

        # Dump
        tr_folds.to_csv(outdir/f'{cv_folds}fold_tr_id.csv', index=False)
        vl_folds.to_csv(outdir/f'{cv_folds}fold_vl_id.csv', index=False)

    lg.kill_logger()
    # print('Done.')
