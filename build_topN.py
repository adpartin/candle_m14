#!/usr/bin/env python

import pandas as pd
import argparse
from pathlib import Path
from functools import reduce

# input files
base_data_dir = './data'
response_path = Path('./data/combined_single_response_agg')
cell_cancer_types_map_path = Path('./data/combined_cancer_types')
drug_list_path = Path('./data/drugs_1800')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

filepath = Path(__file__).resolve().parent  # (ap)
SEED = 0  # (ap)


def parse_arguments(model_name=''):
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_n', type=int, default=6,
                        help='Number of cancer types to be included. Default 6')
    parser.add_argument('--drug_descriptor', type=str, default='dragon7',
                        choices=['dragon7', 'mordred'],
                        help='Drug descriptors')
    parser.add_argument('--cell_feature', default='rnaseq',
                        choices=['rnaseq', 'snps'],
                        help='Cell line features')
    parser.add_argument('--cell_feature_subset', default='lincs1000',
                        choices=['lincs1000', 'oncogenes', 'all'],
                        help='Subset of cell line features. Default lincs1000')
    parser.add_argument('--format', default='hdf5',
                        choices=['csv', 'tsv', 'parquet', 'hdf5', 'feather'],
                        help='Dataframe file format. Default hdf5')
    parser.add_argument('--response_type', default='reg',
                        choices=['reg', 'bin'],
                        help='Response type. Regression(reg) or Binary Classification(bin). Default reg')
    parser.add_argument('--labels', action='store_true',
                        help='Contains Cell and Drug label. Default False')
    parser.add_argument('--target', type=str, default='AUC',
                        choices=['AUC', 'IC50', 'EC50', 'EC50se', 'R2fit', 'Einf', 'HS', 'AAC1', 'AUC1', 'DSS1'],
                        help='Response label value. Default AUC')

    args, unparsed = parser.parse_known_args()
    return args, unparsed


def check_file(filepath):
    print("checking {}".format(filepath))
    status = filepath.is_file()
    if status is False:
        print("File {} is not found in data dir.".format(filepath))
    return status


def check_data_files(args):
    filelist = [response_path, cell_cancer_types_map_path, drug_list_path, get_cell_feature_path(args), get_drug_descriptor_path(args)]
    return reduce((lambda x, y: x & y), map(check_file, filelist))


def get_cell_feature_path(args):
    if args.cell_feature_subset == 'all':
        filename = 'combined_{}_data_combat'.format(args.cell_feature)
    else:
        filename = 'combined_{}_data_{}_combat'.format(args.cell_feature, args.cell_feature_subset)
    return Path(base_data_dir, filename)


def get_drug_descriptor_path(args):
    filename = 'combined_{}_descriptors'.format(args.drug_descriptor)
    return Path(base_data_dir, filename)


def build_file_basename(args):
    return "top_{}.res_{}.cf_{}.dd_{}{}".format(args.top_n, args.response_type, args.cell_feature, args.drug_descriptor,
                                                '.labled' if args.labels else '')


def build_filename(args):
    return "{}.{}".format(build_file_basename(args), args.format)


def build_dataframe(args):

    # Identify Top N cancer types
    df_response = pd.read_csv(response_path, sep='\t', engine='c', low_memory=False)

    df_uniq_cl_drugs = df_response[['CELL', 'DRUG']].drop_duplicates().reset_index(drop=True)

    df_cl_cancer_map = pd.read_csv(cell_cancer_types_map_path, sep='\t', header=None, names=['CELL', 'CANCER_TYPE'])
    df_cl_cancer_map.set_index('CELL')

    df_cl_cancer_drug = df_cl_cancer_map.merge(df_uniq_cl_drugs, on='CELL', how='left', sort='true')
    df_cl_cancer_drug['CELL_DRUG'] = df_cl_cancer_drug.CELL.astype(str) + '.' + df_cl_cancer_drug.DRUG.astype(str)

    top_n = df_cl_cancer_drug.groupby(['CANCER_TYPE']).count().sort_values('CELL_DRUG', ascending=False).head(args.top_n)
    top_n_cancer_types = top_n.index.to_list()

    print("Identified {} cancer types: ".format(args.top_n), top_n_cancer_types)

    # Indentify cell lines associated with the target cancer types
    df_cl = df_cl_cancer_drug[df_cl_cancer_drug['CANCER_TYPE'].isin(top_n_cancer_types)][['CELL']].drop_duplicates().reset_index(drop=True)

    # Identify drugs associated with the target cancer type & filtered by drug_list
    df_drugs = df_cl_cancer_drug[df_cl_cancer_drug['CANCER_TYPE'].isin(top_n_cancer_types)][['DRUG']].drop_duplicates().reset_index(drop=True)

    drug_list = pd.read_csv(drug_list_path)['DRUG'].to_list()
    df_drugs = df_drugs[df_drugs['DRUG'].isin(drug_list)].reset_index(drop=True)

    # Filter response by cell lines (4882) and drugs (1779)
    cl_filter = df_cl.CELL.to_list()
    dr_filter = df_drugs.DRUG.to_list()
    target = args.target

    df_response = df_response[df_response.CELL.isin(cl_filter) & df_response.DRUG.isin(dr_filter)][['CELL', 'DRUG', target]].drop_duplicates().reset_index(drop=True)

    if args.response_type == 'bin':
        df_response[target] = df_response[target].apply(lambda x: 0 if x < 0.5 else 1)
        df_response.rename(columns={target: 'Response'}, inplace=True)

    # Join response data with Drug descriptor & RNASeq
    df_rnaseq = pd.read_csv(get_cell_feature_path(args), sep='\t', low_memory=False)
    df_rnaseq = df_rnaseq[df_rnaseq['Sample'].isin(cl_filter)].reset_index(drop=True)

    df_rnaseq.rename(columns={'Sample': 'CELL'}, inplace=True)
    df_rnaseq.columns = ['GE_' + x if i > 0 else x for i, x in enumerate(df_rnaseq.columns.to_list())]
    df_rnaseq = df_rnaseq.set_index(['CELL'])

    df_descriptor = pd.read_csv(get_drug_descriptor_path(args), sep='\t', low_memory=False, na_values='na')
    df_descriptor = df_descriptor[df_descriptor.DRUG.isin(dr_filter)].set_index(['DRUG']).fillna(0)

    df = df_response.merge(df_rnaseq, on='CELL', how='left', sort='true')
    df.set_index(['DRUG'])
    
    df_final = df.merge(df_descriptor, on='DRUG', how='left', sort='true')
    if args.labels:
        df_cell_map = df_final['CELL'].to_dict()
        df_drug_map = df_final['DRUG'].to_dict()
        df_final.drop(columns=['CELL', 'DRUG'], inplace=True)
        df_final.drop_duplicates(inplace=True)
        df_final.insert(0, 'DRUG', df_final.index.map(df_drug_map))
        df_final.insert(0, 'CELL', df_final.index.map(df_cell_map))
        df_final.reset_index(drop=True, inplace=True)
    else:
        df_final.drop(columns=['CELL', 'DRUG'], inplace=True)
        df_final.drop_duplicates(inplace=True)
    print("Dataframe is built with total {} rows.".format(len(df_final)))

    # Shuffle
    df_final = df_final.sample(frac=1.0, random_state=SEED).reset_index(drop=True) # (ap)    
    
    save_filename = build_filename(args)
    print("Saving to {}".format(save_filename))
    
    if args.format == 'feather':
        df_final.to_feather(save_filename)
    elif args.format == 'csv':
        df_final.to_csv(save_filename, float_format='%g', index=False)
    elif args.format == 'tsv':
        df_final.to_csv(save_filename, sep='\t', float_format='%g', index=False)
    elif args.format == 'parquet':
        df_final.to_parquet(save_filename, index=False)
    elif args.format == 'hdf5':
        df_cl.to_csv(build_file_basename(args) + '_cellline.txt', header=False, index=False)
        df_drugs.to_csv(build_file_basename(args) + '_drug.txt', header=False, index=False)
        df_final.to_hdf(save_filename, key='df', mode='w', complib='blosc:snappy', complevel=9)

    
    # (ap) create outdir
    import os
    args = vars(args)
    outdir = Path('top' + str(args['top_n']) + '_data')
    os.makedirs(outdir, exist_ok=True)
    
    
    # --------------------------------------------------
    # (ap) tissue type histogram
    # --------------------------------------------------
    dd = df_cl_cancer_drug[['CELL', 'DRUG', 'CANCER_TYPE']].merge(
        df_final[['CELL', 'DRUG', 'AUC']], on=['CELL', 'DRUG'], how='inner')
    dd = pd.DataFrame(dd['CANCER_TYPE'].value_counts())
    dd = dd.reset_index().rename(columns={'index': 'ctype', 'CANCER_TYPE': 'count'})
    dd['ctype'] = dd['ctype'].map(lambda x: ' '.join(x.split('_')))
        
    x = dd['ctype']
    y = dd['count']
    ax = dd.plot.barh(x='ctype', y='count', xlim=[0, y.max()*1.15], legend=False, figsize=(7, 5), fontsize=12)
    ax.set_ylabel(None, fontsize=14);
    ax.set_xlabel('Total responses', fontsize=14);
    ax.set_title('Number of AUC responses per cancer type (Top6)', fontsize=14);
    ax.invert_yaxis()

    for p in ax.patches:
        val = int(p.get_width()/1000)
        x = p.get_x() + p.get_width() + 1000
        y = p.get_y() + p.get_height()/2
        ax.annotate(str(val) + 'k', (x, y), fontsize=12)

    # OR
    # fig, ax = plt.subplots(figsize=(7, 5))
    # plt.barh(dd['CANCER_TYPE'], dd['CELL_DRUG'], color='b', align='center', alpha=0.7)
    # plt.xlabel('Total responses', fontsize=14);
    plt.savefig(outdir/'Top{}_histogram.png'.format(args['top_n']), dpi=300, bbox_inches='tight')
    # --------------------------------------------------
    

    # --------------------------------------------------
    # (ap) break data into
    # --------------------------------------------------
    # Split features and traget
    print('\nSplitting features and target ...')
    
    meta = df_final[['AUC', 'CELL', 'DRUG']]
    xdata = df_final.drop(columns=['AUC', 'CELL', 'DRUG'])

    xdata.to_parquet( outdir/'xdata.parquet' )
    meta.to_parquet( outdir/'meta.parquet' )
    # --------------------------------------------------
    
    
    # --------------------------------------------------
    # (ap) generate train/val/test splits
    # --------------------------------------------------
    from data_split import make_split
    print('\nSplitting train/val/test ...')
    args['cell_fea'] = 'GE'
    args['drug_fea'] = 'DD'
    args['te_method'] = 'simple'
    args['cv_method'] = 'simple'
    args['te_size'] = 0.1
    args['vl_size'] = 0.1
    args['n_jobs'] = 4
    make_split(xdata=xdata, meta=meta, outdir=outdir, args=args)
    print('Done.')
    # --------------------------------------------------


if __name__ == '__main__':
    FLAGS, unparsed = parse_arguments()
    if check_data_files(FLAGS):
        build_dataframe(FLAGS)
