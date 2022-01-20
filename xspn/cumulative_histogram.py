from collections import namedtuple
import bisect

import numpy as np

from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import MetaType, Type
from spn.algorithms.Inference import add_node_likelihood


def _check_densities(densities: list[float], eps: float=1e-3):
    if len(densities) < 2:
        return False

    bounded = all(0 <= d and d <= 1 + eps for d in densities)
    ascending = all(a <= b + eps for a, b in zip(densities, densities[1:]))
    endings = abs(densities[0]) <= eps and abs(densities[-1] - 1) <= eps

    if not ascending:
        for a, b in zip(densities, densities[1:]):
            if a > b:
                print(f'{a} <= {b} is violated!')


    return bounded and ascending and endings


class CumulativeHistogram(Leaf):
    type = Type.CATEGORICAL
    property_type = namedtuple("CumulativeHistogram", "densities")

    def __init__(self, densities, scope):
        Leaf.__init__(self, scope=scope)
        self.densities = densities
        self.meta_type = MetaType.DISCRETE

        if not _check_densities(densities):
            raise ValueError(f"densities does not fulfill the required properties: {densities}")


def get_cumulative_histogram_to_str(scope_to_attributes):
    def func(node, feature_names, node_to_str):
        scope = ','.join(scope_to_attributes(node.scope))
        probabilities = ','.join(str(d) for d in node.densities)
        return f'{str(node)} CumulativeHistogram({scope}|[{probabilities}])\n'

    return func


def accumulate_probabilities(probabilities: list[float]):
    densities = [0]

    for p in probabilities:
        densities.append(densities[-1] + p)


    return densities


# inference algorithms

def _cumulative_histogram_lh(densities: list[float], data: np.ndarray, **kwargs):
    probs = np.zeros((data.shape[0], 1))

    for i, x in enumerate(data):
        idx = np.clip(x, 0, len(densities) - 1).astype(np.int64)

        if 'verbose' in kwargs and kwargs['verbose'] == True:
            print(f'_cumulative_histogram_lh idx={idx}')


        probs[i] = densities[idx]


    return probs


def _cumulative_histogram_llh(node: CumulativeHistogram, data=None, dtype=np.float64, **kwargs):
    probs = np.ones((data.shape[0], 1), dtype=dtype)

    nd = data[:, node.scope[0]]
    marg_ids = np.isnan(nd)

    probs[~marg_ids] = _cumulative_histogram_lh(np.array(node.densities), nd[~marg_ids], **kwargs)

    return np.log(probs)


def add_cumulative_histogram_inference_support():
    add_node_likelihood(CumulativeHistogram, log_lambda_func=_cumulative_histogram_llh)
