from collections import namedtuple
import bisect

import numpy as np

from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import MetaType, Type
from spn.algorithms.Inference import add_node_likelihood


def _check_densities(densities):
    if len(densities) < 2:
        return False

    bounded = all(0 <= d and d <= 1 for d in densities)
    ascending = all(a <= b for a, b in zip(densities, densities[1:]))

    if not ascending:
        for a, b in zip(densities, densities[1:]):
            if a > b:
                print(f'{a} <= {b} is violated!')


    return bounded and ascending


class SummedHistogram(Leaf):
    type = Type.CATEGORICAL
    property_type = namedtuple("Histogram", "breaks densities bin_repr_points")

    def __init__(self, breaks, densities, scope):
        Leaf.__init__(self, scope=scope)
        self.breaks = breaks
        self.densities = densities
        self.meta_type = MetaType.DISCRETE

        if not _check_densities(densities):
            raise ValueError(f"densities does not fulfill the required properties: {densities}")


def get_histogram_to_str(scope_to_attributes):
    def func(node, feature_names, node_to_str):
        scope = ','.join(scope_to_attributes(node.scope))
        breaks = ','.join(f'{b}.' for b in node.breaks)
        probabilities = ','.join(str(d) for d in node.densities)
        # we name is Histogram to be compatible with XSPN
        return f'{str(node)} Histogram({scope}|[{breaks}];[{probabilities}])\n'

    return func


def accumulate_probabilities(probabilities):
    densities = [0]

    for p in probabilities:
        densities.append(densities[-1] + p)


    return densities


# inference algorithms

def _summed_histogram_lh(breaks, densities, data, **kwargs):
    probs = np.zeros((data.shape[0], 1))

    for i, x in enumerate(data):
        # probability is 0
        if x < breaks[0]:
            continue

        # probability is 1
        if x >= breaks[-1]:
            probs[i] = 1
            continue

        # the original SPFlow histogram implementation subtracts 1 here
        idx = bisect.bisect(breaks, x)

        if 'verbose' in kwargs and kwargs['verbose'] == True:
            print(f'_summed_histogram_lh idx={idx}')


        probs[i] = densities[idx]


    return probs


def _summed_histogram_llh(node, data=None, dtype=np.float64, **kwargs):
    probs = np.ones((data.shape[0], 1), dtype=dtype)

    nd = data[:, node.scope[0]]
    marg_ids = np.isnan(nd)

    probs[~marg_ids] = _summed_histogram_lh(np.array(node.breaks), np.array(node.densities), nd[~marg_ids], **kwargs)

    return np.log(probs)


def add_summed_histogram_inference_support():
    add_node_likelihood(SummedHistogram, log_lambda_func=_summed_histogram_llh)
