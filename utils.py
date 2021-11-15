#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import sys

import numpy as np

logger = logging.getLogger("dpc_cluster")


def load_paperdata(distance_f):
    """
    Load distance from data

    Args:
        distance_f : distance file, the format is column1-index 1, column2-index 2, column3-distance

    Returns:
        distances dict, max distance, min distance, max continues id
    """
    logger.info("PROGRESS: load data")
    distances = {}
    min_dis, max_dis = sys.float_info.max, 0.0
    max_id = 0
    with open(distance_f, 'r') as fp:
        for line in fp:
            x1, x2, d = line.strip().split(' ')
            x1, x2 = int(x1), int(x2)
            max_id = max(max_id, x1, x2)
            dis = float(d)
            min_dis, max_dis = min(min_dis, dis), max(max_dis, dis)
            distances[(x1, x2)] = float(d)
            distances[(x2, x1)] = float(d)
    for i in range(max_id):
        distances[(i, i)] = 0.0
    logger.info("PROGRESS: load end")
    return distances, max_dis, min_dis, max_id


def convert_to_matrix(distances_dict: dict, max_id: int):
    """
    convert distances dict to distances matrix

    Args:
        distances_dict: distances dict
        max_id:  max continues id equal to size

    Returns:
        distances matrix
    """
    logger.info("PROGRESS: convert data")
    distances = np.zeros((max_id, max_id), float)

    # note that that the distances_dict is one based while the matrix is zero based
    for i in range(max_id):
        for j in range(max_id):
            if i == j:
                distances[i, j] = 0
            else:
                distances[i, j] = distances_dict[(i + 1), (j + 1)]
            distances[j, i] = distances[i, j]
    logger.info("PROGRESS: convert end")
    return distances
