#!/usr/bin/env python3

import os
import sys
import glob
import time
import math

import multiprocessing as mp
import pandas as pd
import numpy as np

# from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn import tree

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=3)


class Config:
    '''Prediction'''
    def __init__(
            self,
            input_csv,
            metadata_csv,
            cluster_csv,
            label_csv,
            columns,
            dists):
        self.input_csv = input_csv
        self.metadata_csv = metadata_csv
        self.cluster_csv = cluster_csv
        self.label_csv = label_csv
        self.columns = columns
        self.dists = dists


if __name__ == '__main__':
    # Configuration at on place

    EDIR = '../../evaluation/'
    DDIR = '../../datasets/'
    TRAIN_SIZE = 10000
    NROWS = 1000000
    CFGS = list()

    # Job-IO-Duration
    METRICS = [
        'md_file_create', 'md_file_delete', 'md_mod', 'md_other', 'md_read',
        'read_bytes', 'read_calls', 'write_bytes', 'write_calls']
    COLUMNS = [str(n) + '_' + metric for n in [1, 4] for metric in METRICS]
    CFGS.append(Config(
        input_csv='%s/%s' % (DDIR, '/job_io_duration.csv'),
        metadata_csv='%s/%s' % (DDIR, 'job_metadata.csv'),
        cluster_csv='%s/%s' % (EDIR, 'job_io_duration_clustered.csv'),
        label_csv='%s/%s' % (EDIR, 'job_io_duration_labeled.csv'),
        columns=COLUMNS,
        #dists=list(np.arange(0.1, math.pow(len(COLUMNS), (1/len(COLUMNS))), 0.2))
        dists=[0.03, 0.06, 0.09, 0.1, 0.2, 0.3]
        ))

    # Job-Metrics
    COLUMNS = ['utilization', 'problem_time', 'balance']
    CFGS.append(Config(
        input_csv='%s/%s' % (DDIR, '/job_metrics.csv'),
        metadata_csv='%s/%s' % (DDIR, 'job_metadata.csv'),
        cluster_csv='%s/%s' % (EDIR, 'job_metrics_clustered.csv'),
        label_csv='%s/%s' % (EDIR, 'job_metrics_labeled.csv'),
        columns=COLUMNS,
        #dists=list(np.arange(0.1, math.pow(len(COLUMNS), (1/len(COLUMNS))), 0.2))
        dists=[0.03, 0.06, 0.09, 0.1, 0.2, 0.3]
        ))


    for cfg in CFGS:
    # Read and prepare input
        DATA = pd.read_csv(cfg.input_csv, index_col='jobid', dtype={'jobid':np.int64}, nrows=NROWS)
        print(DATA.head())
        METADATA = pd.read_csv(cfg.metadata_csv, index_col='jobid', dtype={'jobid':np.int64, 'utilization':np.float}, nrows=NROWS)
        print(METADATA.head())
        DATA = pd.merge(DATA, METADATA, left_on='jobid', right_on='jobid')
        print("DATA", DATA.head())


        # Clustering
        # with hierachical algorithm
        # with several distances
        print('Clustering')
        DATA.dropna(inplace=True)
        RES = list()
        for dist in cfg.dists:
            start = time.time()
            '''Prediction'''
            cluster_data = DATA[cfg.columns]
            cluster_data = cluster_data[0:TRAIN_SIZE]
            X = MinMaxScaler().fit_transform(cluster_data)
            model = AgglomerativeClustering(n_clusters=None, distance_threshold=dist)
            model.fit(X)
            y_pred = model.labels_.astype(np.int)
            cluster_data['cluster'] = y_pred
            cluster_data['dist'] = dist
            RES.append(cluster_data)
            stop = time.time()
            print('Duration %f seconds' % (stop - start))

        RES_DF = pd.concat(RES)
        RES_DF.to_csv(cfg.cluster_csv)



        # Classification
        # with decision trees

        if os.path.exists(cfg.label_csv):
            os.remove(cfg.label_csv)

        for dist in cfg.dists:
            print('Classification %f' % (dist))
            RES_DIST_DF = RES_DF[RES_DF['dist'] == dist]
            X = RES_DIST_DF[cfg.columns]
            Y = RES_DIST_DF['cluster']
            CLF = tree.DecisionTreeClassifier()
            CLF = CLF.fit(X, Y)

            LABELS = CLF.predict(DATA[cfg.columns])
            DATA['label'] = LABELS
            DATA['dist'] = dist
            if os.path.exists(cfg.label_csv):
                DATA.to_csv(cfg.label_csv, mode='a', header=False)
            else:
                DATA.to_csv(cfg.label_csv, mode='w', header=True)

