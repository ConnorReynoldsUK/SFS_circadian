import pandas as pd
import numpy as np
import sys
from functools import  reduce
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import random
from sfs_class import SFS_hub

# change to any number 0-9 if not using command line
gene_number = int(sys.argv[0])

def main_gene_selector(main_gene, df, df_copy):
    if main_gene not in df.index:
        df.loc[main_gene] = df_copy.loc[main_gene]
    return df


def cyclic_time(times):
    # this converts any time to a -cosine and sine value
    times = np.asarray(times)
    times = times % 24
    t_cos = -np.cos((2 * np.pi * times.astype('float64') / 24)+(np.pi/2))
    t_sin = np.sin((2 * np.pi * times.astype('float64') / 24)+(np.pi/2))

    return t_cos, t_sin

def phase_ranker(phases, ranks):
    # ranks genes by ryhthmicity q value
    common_genes = [i for i in ranked_genes.index if i in phases.index]
    ranks = ranks.loc[common_genes].sort_values('mean_q').iloc[:15000, 0]
    phases = phases.loc[ranks.index]

    print(ranks)

    return pd.concat((ranks, phases), axis=1)



N_GENES = 40
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# phases = metacycle defined phase bins
# ranked genes = metacycle q values
phases = pd.read_csv('data/phase_bins.csv', index_col=0)
ranked_genes = pd.read_csv('data/full_variations.csv', index_col=0)
ranked_genes = ranked_genes.sort_values('mean_q')



X_train = pd.read_csv('data/X_combo_noyav.csv', index_col=0)
Y_train = [float(i) for i in X_train.columns]
Y_train_cos, Y_train_sin = cyclic_time(Y_train)
Y_train_data = np.concatenate((np.asarray(Y_train_cos).reshape(-1, 1), np.asarray(Y_train_sin).reshape(-1, 1)), axis=1)


X_yav = pd.read_csv('data/X_yav_gav.csv', index_col=0)
Y_yav = [float(i[3:]) for i in X_yav.columns]
Y_yav_cos, Y_yav_sin = cyclic_time(Y_yav)
Y_yav_data = np.concatenate((np.asarray(Y_yav_cos).reshape(-1, 1), np.asarray(Y_yav_sin).reshape(-1, 1)), axis=1)

X_howe = pd.read_csv('data/X_howe_WT.csv', index_col=0)
Y_howe = [float(i)+3.75 for i in X_howe.columns]
Y_howe_cos, Y_howe_sin = cyclic_time(Y_howe)
Y_howe_data = np.concatenate((np.asarray(Y_howe_cos).reshape(-1, 1), np.asarray(Y_howe_sin).reshape(-1, 1)), axis=1)


X_mas = pd.read_csv('data/X_mas_ours.csv', index_col=0)
Y_mas = [float(i) for i in X_mas.columns]
Y_mas_cos, Y_mas_sin = cyclic_time(Y_mas)
Y_mas_data = np.concatenate((np.asarray(Y_mas_cos).reshape(-1, 1), np.asarray(Y_mas_sin).reshape(-1, 1)), axis=1)

# 666 transcriptomes
X_666 = pd.read_csv('data/tpm_1001g_expression.csv', index_col=0).T


# find common genes
combinds = reduce(np.intersect1d, (ranked_genes.index, X_train.index, X_yav.index, X_mas.index,X_howe.index, X_666.index))


ranked_genes = ranked_genes.loc[combinds]
ranked = phase_ranker(phases, ranked_genes)
X_train = X_train.loc[ranked.index]
X_yav = X_yav.loc[ranked.index]
X_mas = X_mas.loc[ranked.index]
X_howe = X_howe.loc[ranked.index]

X_train_copy = X_train.copy()
X_yav_copy = X_yav.copy()
X_mas_copy = X_mas.copy()
X_howe_copy = X_howe.copy()

# core geenes are used as a seed for building the proxy set
core_genes = {'LHY': 'AT1G01060', 'ELF3': 'AT2G25930', 'TOC1': 'AT5G61380', 'CCA1': 'AT2G46830', 'PRR3': 'AT5G60100',
              'PRR7': 'AT5G02810', 'PRR9': 'AT2G46790', 'COR27': 'AT5G42900', 'LUX': 'AT3G46640', 'BOA': 'AT5G59570'}

core_list = [k for k,v in core_genes.items()]

main_gene = core_genes[core_list[gene_number]]
drop = [i for i in core_genes.values() if i != main_gene]


u_phases = np.unique(ranked['phase'])
# number of genes per phase bin used
N_PER_CLUSTER = 15

counts = ranked['phase'].value_counts()
print(counts)
counts = counts.loc[counts > N_PER_CLUSTER].index.values

# we want to make sure the core gene is NOT removed regardless of its rhythmicity rank
keep = []

for i in range(u_phases.shape[0]):
    i_phase = u_phases[i]
    i_ranked = ranked.loc[ranked['phase'] == i_phase]
    keep.append(i_ranked.index[:N_PER_CLUSTER])

keep = np.concatenate(keep)

X_train = X_train.loc[keep]
X_yav = X_yav.loc[keep]
X_howe = X_howe.loc[keep]
X_mas = X_mas.loc[keep]

X_train = main_gene_selector(main_gene, X_train, X_train_copy)
X_yav = main_gene_selector(main_gene, X_yav, X_yav_copy)
X_howe = main_gene_selector(main_gene, X_howe, X_howe_copy)
X_mas = main_gene_selector(main_gene, X_mas, X_mas_copy)



# normalize all datasets
scaler = StandardScaler()
X_train = pd.DataFrame(data=scaler.fit_transform(X_train.T), index=X_train.columns, columns=X_train.index)
X_yav = pd.DataFrame(data=scaler.transform(X_yav.T), index=X_yav.columns, columns=X_yav.index)
X_howe = pd.DataFrame(data=scaler.fit_transform(X_howe.T), index=X_howe.columns, columns=X_howe.index)
X_mas = pd.DataFrame(data=scaler.fit_transform(X_mas.T), index=X_mas.columns, columns=X_mas.index)
ranked = ranked.loc[X_train.columns]


sfs_i = SFS_hub(rhythmic_scores=ranked, error_threshold=60)
sfs_i.sfs_iterator(X_train, Y_train_data, N_GENES, X_yav, Y_yav_data, main_gene)
# SequentialFeatureSelectionCoreGene(X_train, Y_train_data, ranked, N_GENES, 120, X_yav, Y_yav_data, main_gene, main_gene_idx)
