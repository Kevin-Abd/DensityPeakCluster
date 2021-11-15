#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import math

import numpy as np

logger = logging.getLogger("dpc_cluster")


def select_dc(max_id, max_dis, min_dis, distances, auto=False):
    """
    Select the local density threshold, default is the method used in paper, auto is `auto_select_dc`

    Args:
        max_id    : max continues id
        max_dis   : max distance for all points
        min_dis   : min distance for all points
        distances : distance dict
        auto      : use auto dc select or not

    Returns:
        dc that local density threshold
    """
    logger.info("PROGRESS: select dc")
    if auto:
        return auto_select_dc(max_id, max_dis, min_dis, distances)
    percent = 2.0
    position = int(max_id * (max_id + 1) / 2 * percent / 100)
    dc = sorted(distances.values())[position * 2 + max_id]
    logger.info("PROGRESS: dc - " + str(dc))
    return dc


def auto_select_dc(max_id, max_dis, min_dis, distances):
    """
    Auto select the local density threshold that let average neighbor is 1-2 percent of all nodes.

    Args:
        max_id    : max continues id
        max_dis   : max distance for all points
        min_dis   : min distance for all points
        distances : distance dict

    Returns:
        dc that local density threshold
    """
    dc = (max_dis + min_dis) / 2

    while True:
        nneighs = sum([1 for v in distances.values() if v < dc]) / max_id ** 2
        if nneighs >= 0.01 and nneighs <= 0.002:
            break
        # binary search
        if nneighs < 0.01:
            min_dis = dc
        else:
            max_dis = dc
        dc = (max_dis + min_dis) / 2
        if max_dis - min_dis < 0.0001:
            break
    return dc


def local_density(max_id, distances, dc, guass=True, cutoff=False):
    """
    Compute all points' local density

    Args:
        max_id    : max continues id
        distances : distance dict
        dc        : local density threshold
        guass     : use guass func or not(can't use together with cutoff)
        cutoff    : use cutoff func or not(can't use together with guass)

    Returns:
        local density vector that index is the point index that start from 1
    """
    assert guass and cutoff == False and guass or cutoff == True
    logger.info("PROGRESS: compute local density")
    guass_func = lambda dij, dc: math.exp(- (dij / dc) ** 2)
    cutoff_func = lambda dij, dc: 1 if dij < dc else 0
    func = guass and guass_func or cutoff_func
    rho = [-1] + [0] * max_id
    for i in range(1, max_id):
        for j in range(i + 1, max_id + 1):
            rho[i] += func(distances[(i, j)], dc)
            rho[j] += func(distances[(i, j)], dc)
        if i % (max_id / 10) == 0:
            logger.info("PROGRESS: at index #%i" % (i))
    return np.array(rho, np.float32)


def min_distance(max_id, max_dis, distances, rho):
    """
    Compute all points' min distance to the higher local density point(which is the nearest neighbor)

    Args:
        max_id    : max continues id
        max_dis   : max distance for all points
        distances : distance dict
        rho       : local density vector that index is the point index that start from 1

    Returns:
        min_distance vector, nearest neighbor vector
    """
    logger.info("PROGRESS: compute min distance to nearest higher density neigh")
    sort_rho_idx = np.argsort(-rho)
    delta, nneigh = [0.0] + [float(max_dis)] * (len(rho) - 1), [0] * len(rho)
    delta[sort_rho_idx[0]] = -1.
    for i in range(1, max_id):
        for j in range(0, i):
            old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]
            if distances[(old_i, old_j)] < delta[old_i]:
                delta[old_i] = distances[(old_i, old_j)]
                nneigh[old_i] = old_j
        if i % (max_id / 10) == 0:
            logger.info("PROGRESS: at index #%i" % (i))
    delta[sort_rho_idx[0]] = max(delta)
    return np.array(delta, np.float32), np.array(nneigh, np.float32)


class DensityPeakCluster(object):
    def local_density(self, distances, max_dis, min_dis, max_id, dc=None, auto_select_dc=False):
        """
        Just compute local density

        Args:
            max_id          : max continues id
            max_dis         : max distance for all points
            min_dis         : min distance for all points
            distances       : distance dict
            dc              : local density threshold, call select_dc if dc is None
            auto_select_dc  : auto select dc or not

        Returns:
            local density vector, dc
        """
        assert not (dc != None and auto_select_dc)

        if dc == None:
            dc = select_dc(max_id, max_dis, min_dis, distances, auto=auto_select_dc)
        rho = local_density(max_id, distances, dc)
        return rho, dc

    def cluster(self, distances, max_dis, min_dis, max_id, density_threshold, distance_threshold, dc=None,
                auto_select_dc=False):
        """
        Cluster the data

        Args:
            dc                  : local density threshold, call select_dc if dc is None
            density_threshold   : local density threshold for choosing cluster center
            distance_threshold  : min distance threshold for choosing cluster center
            max_id              : max continues id
            max_dis             : max distance for all points
            min_dis             : min distance for all points
            distances           : distance dict
            dc                  : local density threshold, call select_dc if dc is None
            auto_select_dc      : auto select dc or not

        Returns:
            local density vector, min_distance vector, nearest neighbor vector
        """
        assert not (dc != None and auto_select_dc)
        rho, dc = self.local_density(distances, max_dis, min_dis, max_id, dc=dc, auto_select_dc=auto_select_dc)
        delta, nneigh = min_distance(max_id, max_dis, distances, rho)
        logger.info("PROGRESS: start cluster")
        cluster, ccenter = {}, {}  # cl/icl in cluster_dp.m

        for idx, (ldensity, mdistance, nneigh_item) in enumerate(zip(rho, delta, nneigh)):
            if idx == 0: continue
            if ldensity >= density_threshold and mdistance >= distance_threshold:
                ccenter[idx] = idx
                cluster[idx] = idx
            else:
                cluster[idx] = -1

        # assignation
        ordrho = np.argsort(-rho)
        for i in range(ordrho.shape[0] - 1):
            if ordrho[i] == 0: continue
            if cluster[ordrho[i]] == -1:
                cluster[ordrho[i]] = cluster[nneigh[ordrho[i]]]
            if i % (max_id / 10) == 0:
                logger.info("PROGRESS: at index #%i" % (i))

        # halo
        halo, bord_rho = {}, {}
        for i in range(1, ordrho.shape[0]):
            halo[i] = cluster[i]
        if len(ccenter) > 0:
            for idx in ccenter.keys():
                bord_rho[idx] = 0.0
            for i in range(1, rho.shape[0] - 1):
                for j in range(i + 1, rho.shape[0]):
                    if cluster[i] != cluster[j] and distances[i, j] <= dc:
                        rho_aver = (rho[i] + rho[j]) / 2.0
                        if rho_aver > bord_rho[cluster[i]]:
                            bord_rho[cluster[i]] = rho_aver
                        if rho_aver > bord_rho[cluster[j]]:
                            bord_rho[cluster[j]] = rho_aver
            for i in range(1, rho.shape[0]):
                if rho[i] < bord_rho[cluster[i]]:
                    halo[i] = 0
        for i in range(1, rho.shape[0]):
            if halo[i] == 0:
                cluster[i] = - 1

        self.cluster, self.ccenter = cluster, ccenter
        self.distances = distances
        self.max_id = max_id
        logger.info("PROGRESS: ended")
        return rho, delta, nneigh
