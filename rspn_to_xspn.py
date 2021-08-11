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


scope_to_attributes = lambda x: ''


def sum_to_str(node, feature_names, node_to_str):
    w = node.weights
    ch = node.children
    sumw = ", ".join(map(lambda i: "%s*%s" % (w[i], ch[i].name), range(len(ch))))
    child_str = "".join(map(lambda c: spn_to_str_ref_graph(c, feature_names, node_to_str), node.children))
    child_str = child_str.replace("\n", "\n\t")
    return "%s SumNode(%s){\n\t%s}\n" % (str(node), sumw, child_str)


def identity_numeric_leaf_to_str(node, feature_names, node_to_str):
    #unique_vals, probabilities, scope

    vals = '...' #', '.join(str(v) for v in node.unique_vals)
    probs = '...' #', '.join(str(p) for p in node.prob_sum)
    scope = ', '.join(scope_to_attributes(node.scope))
    size = len(node.unique_vals)

    return f'{str(node)} IdentityNumericLeaf([{scope}], size={size})\n'


def histogram_to_str(node, feature_names, node_to_str):
    scope = ','.join(scope_to_attributes(node.scope))
    breaks = ','.join(f'{b}.' for b in node.breaks)
    probabilities = ','.join(str(d) for d in node.densities)
    return f'{str(node)} Histogram({scope}|[{breaks}];[{probabilities}])\n'


node_to_str = {
    Sum: sum_to_str,
    IdentityNumericLeaf: identity_numeric_leaf_to_str,
    Histogram: histogram_to_str
}


def reshape_histogram(alphas, m):
    """
    This function takes a histogram and puts together a condensed histogram of given size.
    """
    betas = []
    n = len(alphas)

    for i in range(m):
        li = n / m * i
        hi = n / m * (i + 1)

        complete_alphas = sum(alphas[math.ceil(li) : math.floor(hi)])
        front_alphas = (math.ceil(li) - li) * alphas[math.floor(li)]

        if hi < n:
            back_alphas = (hi - math.floor(hi)) * alphas[math.floor(hi)]
        else:
            back_alphas = 0

        betas.append(complete_alphas + front_alphas + back_alphas)

    return betas


class ReducedHistogram:
    def __init__(self, old_densities, new_histogram):
        self.old_densities = old_densities
        self.new_histogram = new_histogram


class ConvertedSPN:
    def __init__(self, spn, table, max_histogram_size, converted_domains) -> None:
        self.old_spn = spn
        self.reduced_histograms = dict()
        self.new_spn = self._convert_spn(spn, table, max_histogram_size, converted_domains)
        rebuild_scopes_bottom_up(self.new_spn)

    def _convert_spn(self, node, table: Table, max_histogram_size, converted_domains):
        if isinstance(node, IdentityNumericLeaf):
            size = len(node.unique_vals)
            points = [] # TODO: What's this?

            # TODO: it's always len(size) + 1 == len(node.prob_sum):
            #

            #print(f'size={size} prob size={len(node.prob_sum)}')
            #print(f'prob_sum={node.prob_sum}')
            #if size != len(node.prob_sum) + 1:
            #    print(f'ERRRRRRRROR')


            if size > max_histogram_size:
                #raise Exception('unexpected histogram reshape')
                unscaled_densities = reshape_histogram(node.return_histogram(), max_histogram_size)
                densities = [min(p, 1) for p in accumulate_probabilities(unscaled_densities)]
                breaks = list(range(len(densities) + 1))
            else:
                # TODO: Get the domain for this scope. Then remap the values in unique_values with the help
                # of DomainConverion.convert(...). Those are our breaks. densitites needs to be prepended
                # with 0s and appended with 1s.
                breaks = ([converted_domains[node.scope[0]].convert(uni) for uni in node.unique_vals] +
                          [converted_domains[node.scope[0]].convert(node.unique_vals[-1] + 1)])
                #breaks.append(max(breaks) + 1)
                densities = np.copy(node.prob_sum) #node.return_histogram(copy=True)
                # clip any numerical errors away
                densities[-1] = 1


            #if len(breaks) != len(densities) + 1:
            #    print(f'breaks={breaks} densities={densities}')
            #    print(f'len breaks={len(breaks)} len densities={len(densities)}')
            #    exit()


            #histogram = Histogram(breaks=breaks, densities=densities, bin_repr_points=points, scope=node.scope)
            histogram = SummedHistogram(breaks=breaks, densities=densities, scope=node.scope)
            histogram.id = node.id

            if size > max_histogram_size:
                self.reduced_histograms[node.id] = ReducedHistogram(node.return_histogram(), histogram)

            return histogram
        elif isinstance(node, rspn.structure.base.Sum):
            children = [self._convert_spn(child, table, max_histogram_size, converted_domains) for child in node.children]
            result = Sum(weights=node.weights, children=children)
            result.id = node.id
            return result
        elif isinstance(node, Product):
            children = [self._convert_spn(child, table, max_histogram_size, converted_domains) for child in node.children]
            result = Product(children)
            result.id = node.id
            return result
        else:
            raise Exception(f'Unsupported node type {type(node)}')


# NOTE: For XSPNs we might need to remap the domain to a reduced interval beginning at 0!
class Domain:
    def __init__(self, column, integral):
        self.min = min(column)
        self.max = max(column)
        self.integral = integral

    def sample(self):
        if self.integral:
            return random.randint(self.min, self.max)
        else:
            return random.random() * (self.max - self.min) + self.min

    def sample_reduced(self, new_size):
        s = (self.sample() - self.min) / new_size
        return int(s) if self.integral else s


class DomainConversion:
    def __init__(self, min_value, max_value, offset, stretch_factor, is_intergral):
        self.min_value = min_value
        self.max_value = max_value
        self.offset = offset
        self.stretch_factor = stretch_factor
        self.is_integral = is_intergral

    def convert(self, value):
        if self.is_integral:
            return math.floor((value - self.offset) / self.stretch_factor)
        else:
            return math.floor((value - self.offset) / self.stretch_factor)

    def unconvert(self, value):
        if self.is_integral:
            return math.floor(value * self.stretch_factor) + self.offset
        else:
            return value * self.stretch_factor + self.offset

    def __str__(self):
        if self.is_integral:
            upper = math.floor((self.max_value - self.offset) / self.stretch_factor)
            return f'[{self.min_value}, {self.max_value}] (integral) -> [0, {upper}] (integral)'
        else:
            upper = math.floor((self.max_value - self.offset) / self.stretch_factor)
            return f'[{self.min_value}, {self.max_value}] (real) -> [0, {upper}] (integral)'


def remap_domain(column, max_size, is_integral) -> DomainConversion:
    min_value = min(column)
    max_value = max(column)
    offset = min_value
    # > 1 if the domain is reduced in size
    if is_integral:
        stretch_factor = max((max_value - min_value) / max_size, 1)
    else:
        stretch_factor = (max_value - min_value) / max_size

    return DomainConversion(min_value, max_value, offset, stretch_factor, is_integral)


def remap_data(data: pd.DataFrame, attribute_types: dict):
    domains = { key: remap_domain(data[f'line_item_sanitized.{key}'], 256,
                                  data.dtypes[f'line_item_sanitized.{key}'] == np.int64)
                for key in attribute_types.keys() }

    df = data.copy(deep=True)

    #print(df)

    for attr_name in domains.keys():
        col_name = f'line_item_sanitized.{attr_name}'
        df[col_name] = df[col_name].map(domains[attr_name].convert)

    return df


def my_plot_spn(spn_root):
    def _add_nodes(node, graph: nx.Graph):
        graph.add_node(node.id)
        #print(f'adding node {node.id}')

        if isinstance(node, Sum) or isinstance(node, Product):
            for child in node.children:
                graph.add_edge(node.id, child.id)
                #print(f'adding edge {node.id} -> {child.id}')
                _add_nodes(child, graph)


    g = nx.Graph()
    _add_nodes(spn_root, g)

    plt.figure()
    nx.draw(g)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # actions
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--reduce', action='store_true')
    parser.add_argument('--generate', action='store_true')

    parser.add_argument('--count_queries', default=0, type=int)
    parser.add_argument('--dataset', default='abc', help='Which dataset to be used')
    parser.add_argument('--pickle_path', default='', help='Pickle file path')
    parser.add_argument('--csv_path', default='', help='CSV file path')
    parser.add_argument('--max_histogram_size', default=256, type=int)

    # eval
    parser.add_argument('--sql_queries', default='', type=str, help='path to sql query file')

    args = parser.parse_args()

    if sum(1 for p in [args.plot, args.eval, args.reduce, args.generate] if p) != 1:
        print(f'Exactly one action [plot, eval, reduce, generate] must be selected!')
        exit()

    if args.dataset == 'tpc-h':
        schema = gen_tpc_h_schema(args.csv_path)
        # TODO: Move this to schema loading!
        attribute_types = {
            #'id': 'bigint',
            'orderkey': 'bigint',
            'partkey': 'bigint',
            'suppkey': 'bigint',
            'linenumber': 'smallint',
            'quantity': 'numeric(8, 3)',
            'extendedprices': 'numeric(12, 3)',
            'discount': 'numeric(8, 3)',
            'tax': 'numeric(8, 3)',
            'returnflag': 'smallint',
            'linestatus': 'smallint',
            'shipdate': 'bigint',
            'commitdate': 'bigint',
            'receiptdate': 'bigint',
            'shipinstruct': 'smallint',
            'shipmode': 'smallint'
        }
    else:
        raise ValueError('Dataset unknown')

    if len(schema.tables) != 1:
        raise NotImplementedError('Conversion of RSPNs for more than 1 table is currently not supported!')

    with open(args.pickle_path, 'rb') as f:
        pkl = pickle.load(f)

        for i, spn in enumerate(pkl.spns):
            scope_to_attributes = lambda scope: [schema.tables[0].attributes[s] for s in scope]
            node_to_str[SummedHistogram] = get_histogram_to_str(scope_to_attributes)

            if True:
                data: pd.DataFrame = read_table_csv(schema.tables[0], csv_seperator=';')
                attribute_ranges = { key: remap_domain(data[f'line_item_sanitized.{key}'], args.max_histogram_size, data.dtypes[f'line_item_sanitized.{key}'] == np.int64)
                                        for key in attribute_types.keys() }
                attributes = schema.tables[0].attributes[0:]
                converted_domains = dict()

                for i, attr_name in enumerate(attributes):
                    attr_range = attribute_ranges[attr_name]
                    #print(f'{i} {attr_name} {attr_range}')
                    converted_domains[i] = attr_range

                df = remap_data(data, attribute_types)
                #print(df)

            #print(f'converted_domains={converted_domains}')

            mspn = spn.mspn
            converted = ConvertedSPN(mspn, schema.tables[0], args.max_histogram_size, converted_domains=converted_domains)

            if args.reduce:
                print(spn_to_str_ref_graph(converted.new_spn, node_to_str=node_to_str))
                attrs = ';'.join(s for s in scope_to_attributes(converted.new_spn.scope))
                print(f'# {attrs}')

                #plot_spn(converted.new_spn, 'plot.png')
                #my_plot_spn(converted.new_spn)

            elif args.generate:
                data: pd.DataFrame = read_table_csv(schema.tables[0], csv_seperator=';')
                attribute_ranges = { key: Domain(data[f'line_item_sanitized.{key}'], data.dtypes[f'line_item_sanitized.{key}'] == np.int64)
                                     for key in attribute_types.keys() }
                attributes = schema.tables[0].attributes[1:]

                for _ in range(args.count_queries):
                    random.shuffle(attributes)
                    n = random.randint(1, 4)
                    attrs = attributes[0:n]
                    op = random.choice(['<='])
                    cond = ' AND '.join(f'{attr} {op} {attribute_ranges[attr].sample()}' for attr in attrs)
                    print(f'SELECT COUNT(*) FROM line_item_sanitized WHERE {cond};')

            elif args.plot:
                #plt.figure()
                n = len(converted.reduced_histograms)
                m = math.ceil(math.sqrt(n))

                for i, (node_id, reduced_histogram) in enumerate(converted.reduced_histograms.items()):
                    old_size = len(reduced_histogram.old_densities)
                    scope = scope_to_attributes(reduced_histogram.new_histogram.scope)
                    print(f'node {node_id} with scope {scope} was reduced to {args.max_histogram_size} from old size {old_size}')

                    #plt.subplot(m, m, i + 1)

                    # plot old
                    old_probs = reduced_histogram.old_densities
                    old_xs = list(range(len(old_probs)))
                    #plt.bar(x=old_xs, height=old_probs)

                    # plot new
                    new_probs = reduced_histogram.new_histogram.densities
                    new_xs = list(range(len(new_probs)))
                    #plt.bar(x=new_xs, height=new_probs)

                    print(f'old={sum(old_probs)} new={sum(new_probs)}')

                #plt.show()

            elif args.eval:
                # parse count queries and input them to SPN
                # raise NotImplementedError()

                print('INFO: adding summed histogram support')
                add_summed_histogram_inference_support()

                with open(args.sql_queries, 'r') as sql_queries:
                    for line in sql_queries.readlines():
                        query = parse_query(line.strip(), schema)
                        assert query.query_type == QueryType.CARDINALITY

                        # TODO: First try RSPN cardinality estimation
                        # TODO: Next try to do cardinality estimation in SPFlow with normal bottom-up evaluation

                        #print(line.strip())
                        exp = estimate_expectation(converted.old_spn, converted.new_spn, schema, line, converted_domains)
                        #print(f'exp={exp} n={exp * 6001215}')

                        #exit()
