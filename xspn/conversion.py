import itertools
from numpy.lib.arraysetops import unique
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
from rspn.structure.leaves import IdentityNumericLeaf, Categorical
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

from xspn.cumulative_histogram import CumulativeHistogram, get_cumulative_histogram_to_str


scope_to_attributes = lambda scope: []


def _sum_to_str(node, feature_names, node_to_str):
    w = node.weights
    ch = node.children
    sumw = ", ".join(map(lambda i: "%s*%s" % (w[i], ch[i].name), range(len(ch))))
    child_str = "".join(map(lambda c: spn_to_str_ref_graph(c, feature_names, node_to_str), node.children))
    child_str = child_str.replace("\n", "\n\t")
    return "%s SumNode(%s){\n\t%s}\n" % (str(node), sumw, child_str)


def _identity_numeric_leaf_to_str(node, feature_names, node_to_str):
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


def _is_ascending(arr: list[int]) -> bool:
    return all(x + 1 == x_next for x, x_next in zip(arr, arr[1:]))


def _make_ascending(unique_vals: list[int], densities: list[float]) -> tuple[list[int], list[float]]:
    new_vals = []
    new_densities = []

    for v, v_next, d, d_next in zip(unique_vals, unique_vals[1:], densities, densities[1:]):
        prob = d_next - d
        v_delta = v_next - v

        #print(f'v_delta={v_next} - {v}')

        new_vals.append(v)
        new_densities.append(d)

        for i in range(1, v_delta):
            new_vals.append(v + i)
            new_densities.append(d_next) # + prob / v_delta * i)


    new_vals.append(unique_vals[-1])

    assert _is_ascending(new_vals)

    return new_vals, new_densities


def _crude_expand(unique_vals: list[int], densities: list[float], size: int) -> tuple[list[int], list[float]]:
    # [0, ..., max_value]

    lower_densities_count = unique_vals[0]
    upper_densities_count = size - (len(densities) + lower_densities_count - 1)

    zeros = [0] * lower_densities_count
    ones = [1] * upper_densities_count
    new_densities = zeros + densities + ones

    breaks = list(range(len(new_densities) + 1))

    return breaks, new_densities


def _expand_unique_vals_and_densities(vals: np.ndarray, densities: np.ndarray, size: int) -> tuple[list[int], list[float]]:  
    # The vals array must be expanded to the form [0, 1, ..., max_value]!

    # first create pythonic lists
    vals = vals.tolist()
    densities = densities.tolist()

    # check if vals has the form [a, a + 1, a + 2, ...]
    if not _is_ascending(vals):
        vals, densities = _make_ascending(vals, densities)

    return _crude_expand(vals, densities, size)


def _fill_values_and_densities(vals: np.ndarray, densities: np.ndarray, size: int):
    vals_list = list(vals)
    spaces = [b - a for a, b in zip(vals_list, vals_list[1:])]

    probs = list(itertools.chain.from_iterable([[density] * l for density, l in zip(densities[1:], spaces)]))
    pre_probs = [0] * (vals_list[0] + 1)
    post_probs = [1] * (size - 1 - vals_list[-1])
    total_probs = pre_probs + probs + post_probs

    if len(total_probs) != size:
        print(f'{vals.shape} vals={vals}')
        print(f'{densities.shape} densities={densities}')
        print(f'{len(spaces)} spaces={spaces}')
        print(f'{len(probs)} probs={probs}')
        print(f'{len(pre_probs)} pre_probs={pre_probs}')
        print(f'{len(post_probs)} post_probs={post_probs}')
        print(f'{len(total_probs)} total_probs={total_probs}')

        raise Exception()


    return total_probs


def _rspn_to_xspn_simple(node, size: int, verbose=False):
    if isinstance(node, IdentityNumericLeaf):       
        # densities encodes P(X < t) for some value t
        #print(f'{node.unique_vals}: {node.unique_vals.shape}')
        #print(f'{node.prob_sum}: {node.prob_sum.shape}')
        #print(node.unique_vals.shape)
        #print(node.prob_sum.shape)

        #print(f'{node.unique_vals.shape} node.unique_vals={node.unique_vals}')
        unique_vals = node.unique_vals.astype(int)
        #breaks, densities = _expand_unique_vals_and_densities(unique_vals, node.prob_sum, 255)
        
        if unique_vals.shape[0] > size:
            raise Exception('unique_vals.shape[0] is too large!')
        
        densities = _fill_values_and_densities(unique_vals, node.prob_sum, size)

        #assert len(densities) == 256

        try:
            histogram = CumulativeHistogram(densities=densities, scope=node.scope)
        except Exception as e:
            print(f'unique_vals={unique_vals} shape={unique_vals.shape}')
            print(f'prob_sum={node.prob_sum} shape={node.prob_sum.shape}')
            raise e


        histogram.id = node.id

        return histogram
    elif isinstance(node, rspn.structure.base.Sum):
        children = [_rspn_to_xspn_simple(child, size, verbose) for child in node.children]
        result = Sum(weights=node.weights, children=children)
        result.id = node.id
        return result
    elif isinstance(node, Product):
        children = [_rspn_to_xspn_simple(child, size, verbose) for child in node.children]
        result = Product(children)
        result.id = node.id
        return result
    elif isinstance(node, Categorical):
        if node.p.shape[0] > 256:
            print(f'{node.scope}: {node.cardinality} {node.p.shape} {node.p}')
        return node
    else:
        raise Exception(f'Unsupported node type {type(node)}')


def rspn_to_xspn_simple(node, size: int):
    xspn = _rspn_to_xspn_simple(node, size, True)
    rebuild_scopes_bottom_up(xspn)

    return xspn


def xspn_to_str(xspn, attributes: list[str]):
    scope_to_attributes = lambda scope: [attributes[s] for s in scope]
    node_to_str[CumulativeHistogram] = get_cumulative_histogram_to_str(scope_to_attributes)
    attrs = ';'.join(s for s in scope_to_attributes(xspn.scope))

    return f'{spn_to_str_ref_graph(xspn, node_to_str=node_to_str)}\n# {attrs}'


def rspn_to_str(xspn, attributes: list[str]):
    scope_to_attributes = lambda scope: [attributes[s] for s in scope]
    attrs = ';'.join(s for s in scope_to_attributes(xspn.scope))

    return f'{spn_to_str_ref_graph(xspn, node_to_str=node_to_str)}\n# {attrs}'
