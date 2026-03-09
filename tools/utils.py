# coding:utf-8
import numpy as np
import yaml
import collections
import os


def parse_config(config):
    """
    Load iGibson config from a dict or a YAML file path.
    Args:
        config: A dict or a string path to a YAML config file.
    Returns:
        A config dictionary.
    """
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    if isinstance(config, collectionsAbc.Mapping):
        return config
    else:
        assert isinstance(config, str)

    if not os.path.exists(config):
        raise IOError(
            "config path {} does not exist. "
            "Please either pass in a dict or a string that represents the file path to the config yaml.".format(
                config
            )
        )
    with open(config, "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data


def prep_masks(n, cortices):
    """
    Create the intra- and inter-cortical matrices, and the list of events

    n:        (numpy.ndarray) The connectome matrix in question
    cortices: (List) A list of lists of `[start, end]` for each cortex

    return G1:     (numpy.ndarray) The intra-cortical connection matrix
    return G2:     (numpy.ndarray) The inter-cortical connection matrix
    return events: (list) A list of lambda functions,
    one corresponding to each neuron in the connectome
    """
    cortex_mask = np.zeros_like(n)
    for i, cortex in enumerate(cortices):
        cortex_mask[cortex[0]:cortex[1], cortex[0]:cortex[1]] += i + 1
    G1 = n.copy()
    G1[cortex_mask == 0] = 0
    G2 = n.copy()
    G2[cortex_mask != 0] = 0

    return G1, G2


def divide(matrix, cortices):
    G1, G2 = prep_masks(matrix, cortices)
    n1 = np.count_nonzero(G1, axis=1)
    n1[n1 == 0] = 1
    n2 = np.count_nonzero(G2, axis=1)
    n2[n2 == 0] = 1

    return G1, G2, n1, n2


def section(sequence):
    starts = np.where(np.diff(sequence) == 1)[0] + 1
    ends = np.where(np.diff(sequence) == -1)[0]

    # 处理最后一段
    if sequence[0] == 1:
        starts = np.concatenate([[0], starts])
    if sequence[-1] == 1:
        ends = np.concatenate([ends, [len(sequence) - 1]])

    # # 获取每段连续时刻的中位数 index
    # medians = [(start + end) // 2 for start, end in zip(starts, ends)]
    # medians = np.array(medians) + 1

    return starts


def link_section(vector, window_size):
    result_vector = np.copy(vector)
    indices_of_ones = np.where(result_vector == 1)[0]

    for i in range(len(indices_of_ones) - 1):
        start_index = indices_of_ones[i]
        end_index = indices_of_ones[i + 1]

        if end_index - start_index - 1 <= window_size:
            result_vector[start_index + 1:end_index] = 1

    return result_vector