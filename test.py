# -*- coding: utf-8 -*-

import argparse
import pickle
import time
from datetime import timedelta
from logger import Logger
from dataset import Market1501
from evaluate import pairwise_distance, evaluate_all

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--data_dir', default='/sda/hd/datasets/reid_datasets',type=str)
parser.add_argument('--fea_dir', default='./test_feats.pkl',type=str)
parser.add_argument('--output_dir', default='./results', type=str)
args = parser.parse_args()

logger_name = "Testing"
logger = Logger(logger_name, logdir=args.output_dir, rank=0)
logger.info(args)

dataset = Market1501(root=args.data_dir)
start_time = time.monotonic()
with open(args.fea_dir, 'rb') as pkl_file:
    features = pickle.load(pkl_file)
end_time = time.monotonic()
logger.info(f'Total running time of loading features: {timedelta(seconds=end_time - start_time)}')

logger.info("Testing on the target domain (Market1501):")
distmat, query_features, gallery_features = pairwise_distance(features, dataset.query, dataset.gallery)
test_cmc, test_mAP = evaluate_all(query_features, gallery_features, distmat, query=dataset.query, gallery=dataset.gallery, cmc_flag=True)
logger.info("mAP: {:.1%}".format(test_mAP))
for r in [1, 5, 10]:
    logger.info("CMC curve, Rank-{:<3}{:.1%}".format(r, test_cmc['market1501'][r - 1]))
