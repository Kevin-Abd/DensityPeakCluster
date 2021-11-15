#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from plot import *
from utils import load_paperdata, convert_to_matrix


def plot(data, auto_select_dc=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    distances_dict, max_dis, min_dis, max_id = load_paperdata(data)
    distances_matrix = convert_to_matrix(distances_dict, max_id)
    dpcluster = DensityPeakCluster()
    rho, rc = dpcluster.local_density(distances_matrix, max_dis, min_dis, max_id, auto_select_dc=auto_select_dc)
    delta, nneigh = min_distance(max_id, max_dis, distances_matrix, rho)
    plot_rho_delta(rho, delta)  # plot to choose the threshold


if __name__ == '__main__':
    # plot('./data/data_in_paper/example_distances.dat')
    # plot('./data/data_others/spiral_distance.dat')
    # plot('./data/data_others/aggregation_distance.dat')
    # plot('./data/data_others/flame_distance.dat')
    plot('./data/data_others/jain_distance.dat')
