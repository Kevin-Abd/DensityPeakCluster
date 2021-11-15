#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from plot import *
from utils import load_paperdata, convert_to_matrix


def plot(data, density_threshold, distance_threshold, auto_select_dc=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    distances_dict, max_dis, min_dis, max_id = load_paperdata(data)
    distances_matrix = convert_to_matrix(distances_dict, max_id)

    dpcluster = DensityPeakCluster()
    rho, delta, nneigh, cluster, ccenter = dpcluster.cluster(distances_matrix, max_dis, min_dis, max_id,
                                                             density_threshold, distance_threshold,
                                                             auto_select_dc=auto_select_dc)
    unique_centers = np.unique(ccenter)
    logger.info(str(len(unique_centers)) + ' center as below')
    for idx, center in np.ndenumerate(unique_centers):
        logger.info('%d %d %f %f' % (idx[0], center, rho[center + 1], delta[center + 1]))
    plot_rho_delta(rho, delta)  # plot to choose the threshold
    plot_rhodelta_rho(rho, delta)
    plot_cluster(distances_matrix, max_id, cluster)


if __name__ == '__main__':
    plot('./data/data_in_paper/example_distances.dat', 20, 0.1, False)
    # plot('./data/data_others/spiral_distance.dat',8,5,False)
    # plot('./data/data_others/aggregation_distance.dat',15,4.5,False)
    # plot('./data/data_others/flame_distance.dat',4,7,False)
    # plot('./data/data_others/jain_distance.dat',12,10,False)
