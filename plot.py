#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from sklearn import manifold

from cluster import *
from plot_utils import *
from utils import convert_to_matrix, load_paperdata


def plot_rho_delta(rho, delta):
    """
    Plot scatter diagram for rho-delta points

    Args:
        rho   : rho list
        delta : delta list
    """
    logger.info("PLOT: rho-delta plot")
    plot_scatter_diagram(0, rho[1:], delta[1:], x_label='rho', y_label='delta', title='Decision Graph')
    plt.show()
    plt.savefig('Decision Graph.jpg')


def plot_cluster(distances, max_id, cluster):
    """
    Plot scatter diagram for final points that using multi-dimensional scaling for data

    Args:
        max_id: matrix size
        distances: distance matrix
        cluster : DensityPeakCluster object
    """
    logger.info("PLOT: cluster result, start multi-dimensional scaling")
    dp = np.zeros((max_id, max_id), dtype=np.float32)
    cls = []
    for i in range(1, max_id):
        for j in range(i + 1, max_id + 1):
            dp[i - 1, j - 1] = distances[i - 1, j - 1]
            dp[j - 1, i - 1] = distances[i - 1, j - 1]
        cls.append(cluster[i])
    cls.append(cluster[max_id])
    cls = np.array(cls, dtype=np.float32)
    fo = open(r'./tmp.txt', 'w')
    fo.write('\n'.join(map(str, cls)))
    fo.close()
    # seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(max_iter=200, eps=1e-4, n_init=1, dissimilarity='precomputed')
    dp_mds = mds.fit_transform(dp.astype(np.float64))
    logger.info("PLOT: end mds, start plot")
    plot_scatter_diagram(1, dp_mds[:, 0], dp_mds[:, 1], title='2D Nonclassical Multidimensional Scaling',
                         style_list=cls)
    plt.show()
    plt.savefig("2D Nonclassical Multidimensional Scaling.jpg")


def plot_rhodelta_rho(rho, delta):
    """
    Plot scatter diagram for rho*delta_rho points

    Args:
        rho   : rho list
        delta : delta list
    """
    logger.info("PLOT: rho*delta_rho plot")
    y = rho * delta
    r_index = np.argsort(-y)
    x = np.zeros(y.shape[0])
    idx = 0
    for r in r_index:
        x[r] = idx
        idx += 1
    plt.figure(2)
    plt.clf()
    plt.scatter(x, y)
    plt.xlabel('sorted rho')
    plt.ylabel('rho*delta')
    plt.title("Decision Graph RhoDelta-Rho")
    plt.show()
    plt.savefig('Decision Graph RhoDelta-Rho.jpg')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    distances_dict, max_dis, min_dis, max_id = load_paperdata('./data/data_in_paper/example_distances.dat')
    distances_matrix = convert_to_matrix(distances_dict, max_id)

    # rho, rc = dpcluster.local_density(distances_matrix, max_dis, min_dis, max_id, auto_select_dc=auto_select_dc)
    # plot_rho_delta(rho, delta)   #plot to choose the threshold

    dpcluster = DensityPeakCluster()
    rho, delta, nneigh, cluster, ccenter = dpcluster.cluster(distances_matrix, max_dis, min_dis, max_id,20, 0.1)

    unique_centers = np.unique(ccenter)
    logger.info(str(len(unique_centers)) + ' center as below')
    for idx, center in np.ndenumerate(unique_centers):
        logger.info('%d %d %f %f' % (idx[0], center, rho[center + 1], delta[center + 1]))
    plot_cluster(distances_matrix, max_id, cluster)
