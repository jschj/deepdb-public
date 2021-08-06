"""
This module is a custom rebuild of cardinality estimation which should ease the implementation of a FPGA version.
"""

from pkl_to_spn import scope_to_attributes
from ensemble_compilation.graph_representation import Query, SchemaGraph, Table
from data_preparation.prepare_single_tables import read_table_csv
import pickle
import math
import itertools
import copy
import bisect

from spn.io.Text import spn_to_str_ref_graph
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.histogram.Inference import histogram_log_likelihood
from spn.structure.Base import Product, get_topological_order, Sum
from spn.algorithms.Inference import likelihood, log_likelihood, add_node_likelihood
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

from summed_histogram import SummedHistogram, add_summed_histogram_inference_support


true_leaf_lhs = dict()
custom_leaf_lhs = dict()


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
        merged_dicts = dict()
        product = np.nan
        for child in node.children:
            if len(relevant_scope.intersection(child.scope)) > 0:
                leaf_values, factor = expectation_recursive(child, feature_scope, inverted_features, relevant_scope, evidence,
                                                            node_expectation, node_likelihoods)
                # NaN is treated as 1 (like summing out), NaN is returned if product = factor = NaN
                product = nanproduct(product, factor)
                merged_dicts = dict(itertools.chain(merged_dicts.items(), leaf_values.items()))
        return merged_dicts, product

    elif isinstance(node, rspn.structure.base.Sum):
        if len(relevant_scope.intersection(node.scope)) == 0:
            return np.nan

        child_values = [expectation_recursive(child, feature_scope, inverted_features, relevant_scope, evidence,
                                            node_expectation, node_likelihoods)
                        for child in node.children]
        list_of_leaf_values, llchildren = zip(*child_values)
        merged_dicts = dict(itertools.chain(*[l.items() for l in list_of_leaf_values]))

        #print(merged_dicts)

        # children that are NaN are excluded
        relevant_children_idx = np.where(np.isnan(llchildren) == False)[0]

        if len(relevant_children_idx) == 0:
            return np.nan

        # TODO: Can we not assume that the sum of weights = 1?
        weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
        weighted_sum = sum(node.weights[j] * llchildren[j] for j in relevant_children_idx)

        return merged_dicts, weighted_sum / weights_normalizer

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
        #true_leaf_lhs[node.id] = leaf_lh
        #print(f'    true lh of {node.id} with scope {node.scope} = {leaf_lh}')
        return {node.id: leaf_lh}, leaf_lh


def custom_expectation_recursive(node, evidence, node_likelihoods):
    if isinstance(node, Product):
        child_values = [custom_expectation_recursive(child, evidence, node_likelihoods)
                        for child in node.children]
        #print(*[list(v[0].items()) for v in child_values])
        merged_dicts = dict(itertools.chain(*[list(v[0].items()) for v in child_values]))

        return merged_dicts, math.prod(v[1] for v in child_values)
    elif isinstance(node, Sum):
        weight_sum = sum(node.weights)
        #assert np.isclose(weight_sum, 1)
        child_values = [custom_expectation_recursive(child, evidence, node_likelihoods)
                        for child in node.children]
        #print(*[list(v[0].items()) for v in child_values])
        merged_dicts = dict(itertools.chain(*[list(v[0].items()) for v in child_values]))

        return merged_dicts, sum(weight * child for weight, child in zip(node.weights, [v[1] for v in child_values])) / weight_sum
    elif isinstance(node, Histogram) or isinstance(node, SummedHistogram):
        # convert evidence (in SPFlow format) into format for identity_likelihood_range():
        # replace nan with None and every real number R with NumericRange(-inf, R)

        converted_evidence = np.array([[None] * evidence.shape[1]])
        idx = np.argwhere(~np.isnan(evidence[0]))

        #for to in evidence[0, idx]:
        #    print(to.item())
        #exit()

        converted_evidence[0, idx] = np.array([[NumericRange([[-np.inf, to.item()]]) for to in evidence[0, idx]]]).T

        leaf_lh = node_likelihoods[type(node)](node, converted_evidence).item()
        #custom_leaf_lhs[node.id] = leaf_lh

        #if node.id == 232 or node.id == 20:
        #    print(f'breaks={node.breaks} densities={node.densities}')

        #if not np.isclose(leaf_lh, 1) or node.id == 232:
        #print(f'    custom lh of {node.id} with scope {node.scope} = {leaf_lh}')
        return {node.id: leaf_lh}, leaf_lh
    else:
        raise Exception('unexpected node type')


# TODO: WHAT THE FUCK IS WRONG WITH YOU
def _histogram_interval_probability(node, upper_bound, print_index=False):
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

    #if node.id == 4:
    #    print(f'upper={upper_bound} higher_idx={higher_idx} unique={unique_vals} dens={node.densities}')

    if higher_idx >= len(node.densities):
        return 1

    #if print_index:
    #    print(f'custom index={higher_idx}')

    return node.densities[higher_idx]


def _histogram_likelihood(node, evidence):
    assert len(node.scope) == 1

    probs = np.zeros((evidence.shape[0], 1), dtype=np.float64)
    ranges = evidence[:, node.scope[0]]

    #print(f'ranges={ranges}')

    #if node.id == 4:
    #    print(f'probing node {node.id} with scope {node.scope} and evidence {evidence}')

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

    # check what SPFlow would calculate

    # convert data in SPFlow format
    data = np.array([[(np.nan if interval is None else interval.ranges[0][1]) for interval in evidence[0]]])
    # SPFlows likelihood
    if not np.isnan(data[0][node.scope[0]]):
        val = data[0][node.scope[0]]
        idx = bisect.bisect(node.breaks, val)
        out_of_bounds = val < node.breaks[0] or val >= node.breaks[-1]
    else:
        idx = -1
        out_of_bounds = True

    ll = histogram_log_likelihood(node, data)
    lh = np.exp(ll)

    error = np.abs(probs - lh)

    # if error is too large repeat calculation and compare indices
    if error > 1e-3:
        custom_spflow_prob = node.densities[idx - 1]
        #print(f'custom lh={probs} spflow lh={lh} custom spflow prob={custom_spflow_prob} idx={idx} oob={out_of_bounds} error={np.abs(probs - lh)} density_len={len(node.densities)} densities={node.densities} breaks={node.breaks}')
        #print(f'SPFlow index={idx}')
        _histogram_interval_probability(node, interval[1], True)

        #if not out_of_bounds:
        #    raise Exception("not out_of_bounds detected!")

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
        Histogram: _histogram_likelihood,
        SummedHistogram: _histogram_likelihood
    }

    true_leaf_lhs = dict()
    custom_leaf_lhs = dict()

    print('--------------------------------------------------')

    #lh = likelihood(new_spn, data)
    custom_merged, custom_lh = custom_expectation_recursive(new_spn, data, node_likelihoods=nlhs)
    true_merged, true_lh = expectation(old_spn, table, query_str, schema)
    spflow_lh = likelihood(new_spn, data).item()

    error = abs(custom_lh - true_lh)
    #print(f'query={query_str.strip()}')
    print(f'custom={custom_lh} true={true_lh} spflow={spflow_lh} error={error}')
    #exit()

    #print(f'custom_merged={custom_merged}')
    #print(f'true_merged={true_merged}')

    #if error != 0:
    #    common_keys = set(custom_merged.keys()).intersection(set(true_merged.keys()))
    #    digest = {k: (custom_merged[k], true_merged[k]) for k in common_keys}

    #    for k, v in digest.items():
    #        pass #print(f'{k}: {v[0]} <-> {v[1]}')


    return true_lh
