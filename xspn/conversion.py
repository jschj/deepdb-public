from pkl_to_spn import scope_to_attributes
from ensemble_compilation.graph_representation import Table
from data_preparation.prepare_single_tables import read_table_csv
import pickle
import math

from spn.io.Text import spn_to_str_ref_graph
from spn.structure.leaves.histogram.Histograms import Histogram
import spn.structure as structure
from spn.structure.Base import Product, Sum, rebuild_scopes_bottom_up
from spn.io.Graphics import plot_spn

#from rspn.structure.base import Sum
import rspn.structure.base
from rspn.structure.leaves import IdentityNumericLeaf
from schemas.tpc_h.schema import gen_tpc_h_schema
from schemas.tpc_h_converted.schema import gen_tpc_h_converted_schema
from evaluation.utils import parse_query
from ensemble_compilation.graph_representation import QueryType

import argparse
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from cardinality_custom import estimate_expectation
from summed_histogram import SummedHistogram, get_histogram_to_str, add_summed_histogram_inference_support, accumulate_probabilities

import networkx as nx


scope_to_attributes = lambda scope: []


def _sum_to_str(node, feature_names, node_to_str):
    w = node.weights
    ch = node.children
    sumw = ", ".join(map(lambda i: "%s*%s" % (w[i], ch[i].name), range(len(ch))))
    child_str = "".join(map(lambda c: spn_to_str_ref_graph(c, feature_names, node_to_str), node.children))
    child_str = child_str.replace("\n", "\n\t")
    return "%s SumNode(%s){\n\t%s}\n" % (str(node), sumw, child_str)


def _identity_numeric_leaf_to_str(node, feature_names, node_to_str):
    #unique_vals, probabilities, scope

    vals = '...' #', '.join(str(v) for v in node.unique_vals)
    probs = '...' #', '.join(str(p) for p in node.prob_sum)
    scope = ', '.join(scope_to_attributes(node.scope))
    size = len(node.unique_vals)

    return f'{str(node)} IdentityNumericLeaf([{scope}], size={size})\n'


def _histogram_to_str(node, feature_names, node_to_str):
    scope = ','.join(scope_to_attributes(node.scope))
    breaks = ','.join(f'{b}.' for b in node.breaks)
    probabilities = ','.join(str(d) for d in node.densities)
    return f'{str(node)} Histogram({scope}|[{breaks}];[{probabilities}])\n'


node_to_str = {
    Sum: _sum_to_str,
    IdentityNumericLeaf: _identity_numeric_leaf_to_str,
    Histogram: _histogram_to_str
}


def _rspn_to_xspn_simple(node):
    if isinstance(node, IdentityNumericLeaf):
        breaks = list(node.unique_vals) + [node.unique_vals[-1] + 1]
        breaks.append(breaks[-1] + 1)
        densities = np.copy(node.prob_sum)
        densities[-1] = 1

        histogram = SummedHistogram(breaks=breaks, densities=densities, scope=node.scope)
        histogram.id = node.id

        return histogram
    elif isinstance(node, rspn.structure.base.Sum):
        children = [_rspn_to_xspn_simple(child) for child in node.children]
        result = Sum(weights=node.weights, children=children)
        result.id = node.id
        return result
    elif isinstance(node, Product):
        children = [_rspn_to_xspn_simple(child) for child in node.children]
        result = Product(children)
        result.id = node.id
        return result
    else:
        raise Exception(f'Unsupported node type {type(node)}')


def rspn_to_xspn_simple(node):
    xspn = _rspn_to_xspn_simple(node)
    rebuild_scopes_bottom_up(xspn)

    return xspn


def xspn_to_str(xspn, attributes: list[str]):
    scope_to_attributes = lambda scope: [attributes[s] for s in scope]
    node_to_str[SummedHistogram] = get_histogram_to_str(scope_to_attributes)
    attrs = ';'.join(s for s in scope_to_attributes(xspn.scope))

    return f'{spn_to_str_ref_graph(xspn, node_to_str=node_to_str)}\n# {attrs}'
