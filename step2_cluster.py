#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
from plot import *
from cluster import *
from utils import load_paperdata


def plot(data, density_threshold, distance_threshold, auto_select_dc = False):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	distances, max_dis, min_dis, max_id = load_paperdata(data)

	dpcluster = DensityPeakCluster()
	rho, delta, nneigh = dpcluster.cluster(distances, max_dis, min_dis, max_id, density_threshold, distance_threshold, auto_select_dc = auto_select_dc)
	logger.info(str(len(dpcluster.ccenter)) + ' center as below')
	for idx, center in dpcluster.ccenter.items():
		logger.info('%d %f %f' %(idx, rho[center], delta[center]))
	plot_rho_delta(rho, delta)   #plot to choose the threthold
	plot_rhodelta_rho(rho,delta)
	plot_cluster(dpcluster)


if __name__ == '__main__':
	plot('./data/data_in_paper/example_distances.dat', 20, 0.1,False)
	#plot('./data/data_others/spiral_distance.dat',8,5,False)
	#plot('./data/data_others/aggregation_distance.dat',15,4.5,False)
	#plot('./data/data_others/flame_distance.dat',4,7,False)
	#plot('./data/data_others/jain_distance.dat',12,10,False)