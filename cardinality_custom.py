"""
This module is a custom rebuild of cardinality estimation which should ease the implementation of a FPGA version.
"""

from pkl_to_spn import scope_to_attributes
from ensemble_compilation.graph_representation import Query, SchemaGraph, Table
from data_preparation.prepare_single_tables import read_table_csv
import pickle
import math

from spn.io.Text import spn_to_str_ref_graph
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.Base import Product, get_topological_order, Sum
from spn.algorithms.Inference import likelihood, log_likelihood
from spn.structure.leaves.parametric.Parametric import Categorical

#from rspn.structure.base import Sum
import rspn
from rspn.structure.leaves import IdentityNumericLeaf, identity_likelihood_range
from schemas.tpc_h.schema import gen_tpc_h_schema
from evaluation.utils import parse_query
from ensemble_compilation.graph_representation import QueryType
from rspn.algorithms.ranges import NumericRange

import argparse
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def expectation(spn, table: Table, query_str, schema):
    query: Query = parse_query(query_str, schema)
    # assumes <= conditions only!
    # Histograms are configured such that they return P(X <= val)
    leq_conditions = [cond[1].split('<=') for cond in query.conditions]
    scope = set(table.attributes.index(cond[0]) for cond in leq_conditions)
    # TODO: must be array with None's except where there is evidence (NumericRange)
    
    arr = [[None] * len(table.attributes)]
    evidence = np.array(arr)

    indices = [table.attributes.index(cond[0]) for cond in leq_conditions]

    for i, to in zip(indices, [float(cond[1]) for cond in leq_conditions]):
        evidence[0, i] = NumericRange(ranges=[[-np.inf, to]])

    #print(f'evidence={evidence}')
    #exit()

    #evidence = dict(zip(scope, [cond[1] for cond in leq_conditions]))

    nlhs = {IdentityNumericLeaf: identity_likelihood_range}

    prob = expectation_recursive(node=spn,
                                 feature_scope=[],
                                 inverted_features=[],
                                 relevant_scope=scope,
                                 evidence=evidence,
                                 node_expectation=None,
                                 node_likelihoods=nlhs)

    return prob


def nanproduct(product, factor):
    if np.isnan(product):
        if not np.isnan(factor):
            return factor
        else:
            return np.nan
    else:
        if np.isnan(factor):
            return product
        else:
            return product * factor


def expectation_recursive(node, feature_scope, inverted_features, relevant_scope, evidence, node_expectation,
                          node_likelihoods):
    if isinstance(node, Product):

        product = np.nan
        for child in node.children:
            if len(relevant_scope.intersection(child.scope)) > 0:
                factor = expectation_recursive(child, feature_scope, inverted_features, relevant_scope, evidence,
                                               node_expectation, node_likelihoods)
                # NaN is treated as 1 (like summing out), NaN is returned if product = factor = NaN
                product = nanproduct(product, factor)
        return product

    elif isinstance(node, rspn.structure.base.Sum):
        if len(relevant_scope.intersection(node.scope)) == 0:
            return np.nan

        llchildren = [expectation_recursive(child, feature_scope, inverted_features, relevant_scope, evidence,
                                            node_expectation, node_likelihoods)
                      for child in node.children]

        # children that are NaN are excluded
        relevant_children_idx = np.where(np.isnan(llchildren) == False)[0]

        if len(relevant_children_idx) == 0:
            return np.nan

        # TODO: Can we not assume that the sum of weights = 1?
        weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
        weighted_sum = sum(node.weights[j] * llchildren[j] for j in relevant_children_idx)

        return weighted_sum / weights_normalizer

    else:
        # seems always false in our example
        if node.scope[0] in feature_scope:
            raise Exception('Unexpected branch!')
            t_node = type(node)
            if t_node in node_expectation:

                feature_idx = feature_scope.index(node.scope[0])
                inverted = inverted_features[feature_idx]

                return node_expectation[t_node](node, evidence, inverted=inverted).item()
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        # NOTE: This can be represented by a single value in a histogram!
        leaf_lh = node_likelihoods[type(node)](node, evidence).item()
        #print(f'    true lh of {node.id} with scope {node.scope} = {leaf_lh}')
        return leaf_lh


def custom_expectation_recursive(node, evidence, node_likelihoods):
    if isinstance(node, Product):
        return math.prod(custom_expectation_recursive(child, evidence, node_likelihoods) for child in node.children)
    elif isinstance(node, Sum):
        weight_sum = sum(node.weights)
        #assert np.isclose(weight_sum, 1)

        return sum(weight * custom_expectation_recursive(child, evidence, node_likelihoods)
                   for weight, child in zip(node.weights, node.children)) / weight_sum
    elif isinstance(node, Histogram):
        # convert evidence (in SPFlow format) into format for identity_likelihood_range():
        # replace nan with None and every real number R with NumericRange(-inf, R)

        converted_evidence = np.array([[None] * evidence.shape[1]])
        idx = np.argwhere(~np.isnan(evidence[0]))

        #for to in evidence[0, idx]:
        #    print(to.item())
        #exit()

        converted_evidence[0, idx] = np.array([[NumericRange([[-np.inf, to.item()]]) for to in evidence[0, idx]]]).T

        leaf_lh = node_likelihoods[type(node)](node, converted_evidence).item()

        #if node.id == 232 or node.id == 20:
        #    print(f'breaks={node.breaks} densities={node.densities}')

        #if not np.isclose(leaf_lh, 1) or node.id == 232:
        #    print(f'    custom lh of {node.id} with scope {node.scope} = {leaf_lh}')
        return leaf_lh
    else:
        raise Exception('unexpected node type')


# TODO: WHAT THE FUCK IS WRONG WITH YOU
def _histogram_interval_probability(node, upper_bound):
    """Returns the interval probability of [-inf, upper_bound]."""

    #print(f'densities={node.densities}')
    #print(f'breaks={node.breaks}')

    unique_vals = node.breaks

    if upper_bound == np.inf:
        higher_idx = len(unique_vals)
    else:
        higher_idx = np.searchsorted(unique_vals, upper_bound, side='right')

    #if upper_bound <= 0:
    #    return 0

    #if node.id == 232 or node.id == 20:
    #    print(f'upper={upper_bound} higher_idx={higher_idx} unique={unique_vals} dens={node.densities}')

    if higher_idx >= len(node.densities):
        return 1

    p = node.densities[higher_idx] - node.densities[0]
    return p


def _histogram_likelihood(node, evidence):
    assert len(node.scope) == 1

    probs = np.zeros((evidence.shape[0], 1), dtype=np.float64)
    ranges = evidence[:, node.scope[0]]

    #print(f'ranges={ranges}')

    for i, rang in enumerate(ranges):
        # range == None => sum out!
        if rang is None:
            probs[i] = 1
            continue

        # ignore some other edge cases...

        assert len(rang.get_ranges()) == 1

        interval = rang.get_ranges()[0]
        assert np.isinf(interval[0])

        probs[i] = _histogram_interval_probability(node, interval[1])

        #for k, interval in enumerate(rang.get_ranges()):
        #    inclusive = rang.inclusive_intervals[k]

        #    probs[i] += _histogram_interval_probability(node, interval[0], interval[1], rang.null_value,
        #                                                inclusive[0], inclusive[1])


    return probs


def estimate_expectation(old_spn, new_spn, schema: SchemaGraph, query_str, converted_domains):
    query: Query = parse_query(query_str, schema)
    # assumes <= conditions only!
    # Histograms are configured such that they return P(X <= val)
    leq_conditions = [cond[1].split('<=') for cond in query.conditions]

    #print(leq_conditions)

    # create data in SPFlow format ...
    table: Table = schema.tables[0]
    data = np.empty((1, len(table.attributes)))
    data[:] = np.nan
    indices = [table.attributes.index(cond[0]) for cond in leq_conditions]
    data[0, indices] = [converted_domains[var_index].convert(float(cond[1]))
                        for cond, var_index in zip(leq_conditions, indices)]

    nlhs = {
        IdentityNumericLeaf: identity_likelihood_range,
        Histogram: _histogram_likelihood
    }

    #lh = likelihood(new_spn, data)
    custom_lh = custom_expectation_recursive(new_spn, data, node_likelihoods=nlhs)
    true_lh = expectation(old_spn, table, query_str, schema)

    print(f'custom={custom_lh} true={true_lh}')

    return true_lh
