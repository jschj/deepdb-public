from rspn.structure.leaves import IdentityNumericLeaf
import pandas as pd
import numpy as np
import math

from summed_histogram import SummedHistogram
from xspn.cumulative_histogram import CumulativeHistogram, _cumulative_histogram_llh

from ensemble_compilation.graph_representation import Query, SchemaGraph, Table
from rspn.algorithms.expectations import expectation_recursive, expectation_recursive_batch
from rspn.algorithms.ranges import NumericRange
from rspn.structure.leaves import IdentityNumericLeaf, identity_likelihood_range

# SPFlow stuff
from spn.structure.Base import Product, Sum
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.algorithms.Inference import likelihood



def _histogram_interval_prob(node: SummedHistogram, bound: int) -> float:
    # returns P(x < bound)

    return node.densities[int(bound)]

    unique_vals = node.breaks

    if bound == np.inf:
        higher_idx = len(unique_vals)
    else:
        higher_idx = np.searchsorted(unique_vals, bound, side='right')

    if higher_idx >= len(node.densities):
        return 1

    #print(f'unique_vals={unique_vals} higher_idx={higher_idx} bound={int(bound)}')

    return node.densities[higher_idx]


def _histogram_interval_prob_alt(node: SummedHistogram, bound: int) -> float:
    idx = int(bound) + 1
    
    if idx < 0:
        return 0
    elif idx >= len(node.densities):
        return 1
    else:
        return node.densities[idx]



def _histogram_likelihood(node: SummedHistogram, evidence: np.ndarray) -> np.ndarray:
    assert len(node.scope) == 1
    variable = node.scope[0]
    result = np.array([1 if np.isnan(row[variable]) else _histogram_interval_prob(node, row[variable])
                       for row in evidence])

    return result


def rspn_expectation_recursive(node, evidence: np.ndarray, node_likelihoods: dict) -> np.ndarray:
    if isinstance(node, Product):
        child_values = np.array([rspn_expectation_recursive(child, evidence, node_likelihoods)
                                 for child in node.children])

        return np.prod(child_values, axis=0)
    elif isinstance(node, Sum):
        weight_sum = sum(node.weights)
        #assert np.isclose(weight_sum, 1)

        child_values = np.array([rspn_expectation_recursive(child, evidence, node_likelihoods) * weight
                                 for weight, child in zip(node.weights, node.children)])

        return np.sum(child_values, axis=0) / weight_sum
    elif isinstance(node, SummedHistogram):
        return node_likelihoods[type(node)](node, evidence)
    elif isinstance(node, CumulativeHistogram):
        return node_likelihoods[type(node)](node, evidence)
    else:
        raise Exception('unexpected node type')



def rspn_expectation(spn_root, evidence: np.ndarray) -> np.ndarray:
    nlhs = {
        SummedHistogram: _histogram_likelihood,
        CumulativeHistogram: _cumulative_histogram_llh
    }

    return rspn_expectation_recursive(spn_root, evidence, node_likelihoods=nlhs)


def spflow_expectation(spn_root, evidence: np.ndarray) -> np.ndarray:
    return likelihood(spn_root, evidence)


def original_expectation(spn_root, evidence: np.ndarray) -> np.ndarray:
    # convert data to ranges, one extra entry for id
    ranges = np.array([[None] * (evidence.shape[1] + 1)] * evidence.shape[0])

    for i, row in enumerate(evidence):
        for j, val in enumerate(row, start=0):
            if not np.isnan(val):
                ranges[i, j] = NumericRange(ranges=[[-np.inf, val]])


    nlhs = {
        IdentityNumericLeaf: identity_likelihood_range
    }

    scope_set = set(spn_root.scope)

    #return expectation_recursive(spn_root,
    #    feature_scope=set(),
    #    inverted_features=[],
    #    relevant_scope=scope_set,
    #    evidence=ranges,
    #    node_expectation=[],
    #    node_likelihoods=nlhs)

    return expectation_recursive_batch(spn_root,
        feature_scope=set(),
        inverted_features=[],
        relevant_scope=scope_set,
        evidence=ranges,
        node_expectation=[],
        node_likelihoods=nlhs)


def estimate_expectation(spn, schema: SchemaGraph, query_str: str, sql=True):
    
    
    
    if sql:
        
        
        pass
    else:
        pass

    pass

