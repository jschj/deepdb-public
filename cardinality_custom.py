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
from spn.structure.Base import Product
import spn.structure.Base as spn
from spn.algorithms.Inference import log_likelihood

from rspn.structure.base import Sum
from rspn.structure.leaves import IdentityNumericLeaf
from schemas.tpc_h.schema import gen_tpc_h_schema
from evaluation.utils import parse_query
from ensemble_compilation.graph_representation import QueryType

import argparse
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


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

    elif isinstance(node, Sum):
        if len(relevant_scope.intersection(node.scope)) == 0:
            return np.nan

        llchildren = [expectation_recursive(child, feature_scope, inverted_features, relevant_scope, evidence,
                                            node_expectation, node_likelihoods)
                      for child in node.children]

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

        return node_likelihoods[type(node)](node, evidence).item()


def estimate_expectation(spn, schema: SchemaGraph, query_str):
    query: Query = parse_query(query_str, schema)
    # assumes <= conditions only!
    # Histograms are configured such that they return P(X <= val)
    leq_conditions = [cond[1].split('<=') for cond in query.conditions]

    print(leq_conditions)

    # create data in SPFlow format ...
    table: Table = schema.tables[0]
    data = np.empty((1, len(table.attributes)))
    data[:] = np.nan
    indices = [table.attributes.index(cond[0]) for cond in leq_conditions]
    data[0, indices] = [cond[1] for cond in leq_conditions]

    print(data)

    ll = log_likelihood(spn, data)

    return np.exp(ll)
