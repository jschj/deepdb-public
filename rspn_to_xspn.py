from pkl_to_spn import scope_to_attributes
from ensemble_compilation.graph_representation import Table
from data_preparation.prepare_single_tables import read_table_csv
import pickle
import math

from spn.io.Text import spn_to_str_ref_graph
from spn.structure.leaves.histogram.Histograms import Histogram

from rspn.structure.base import Sum
from rspn.structure.leaves import IdentityNumericLeaf
from schemas.tpc_h.schema import gen_tpc_h_schema

import argparse
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


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
    def __init__(self, old_densities, new_densities):
        self.old_densities = old_densities
        self.new_densities = new_densities


class ConvertedSPN:
    def __init__(self, spn, table, max_histogram_size) -> None:
        self.old_spn = spn
        self.reduced_histograms = dict()
        self.new_spn = self._convert_spn(spn, table, max_histogram_size)

    def _convert_spn(self, node, table: Table, max_histogram_size):
        if isinstance(node, IdentityNumericLeaf):
            size = len(node.unique_vals)
            points = [] # TODO: What's this?

            if size > max_histogram_size:
                densities = reshape_histogram(node.return_histogram(), max_histogram_size)
                breaks = list(range(max_histogram_size + 1))
            else:
                breaks = list(range(size + 1))
                densities = node.return_histogram(copy=True)

            histogram = Histogram(breaks=breaks, densities=densities, bin_repr_points=points, scope=node.scope)
            histogram.id = node.id

            if size > max_histogram_size:
                self.reduced_histograms[node.id] = ReducedHistogram(node.return_histogram(), histogram.densities)

            return histogram
        else:
            for i in range(len(node.children)):
                node.children[i] = self._convert_spn(node.children[i], table, max_histogram_size)

            return node


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='abc', help='Which dataset to be used')
    parser.add_argument('--pickle_path', default='', help='Pickle file path')
    parser.add_argument('--csv_path', default='', help='CSV file path')
    parser.add_argument('--max_histogram_size', default=256, type=int)
    parser.add_argument('--plot', default=False)
    parser.add_argument('--count_queries', default=0, type=int)

    args = parser.parse_args()

    if args.dataset == 'tpc-h':
        schema = gen_tpc_h_schema(args.csv_path)
        # TODO: Move this to schema scheme loading!
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

            mspn = spn.mspn
            converted = ConvertedSPN(mspn, schema.tables[0], args.max_histogram_size)
            #sys.exit()

            #print(spn_to_str_ref_graph(converted.new_spn, node_to_str=node_to_str))
            #attrs = ';'.join(s for s in scope_to_attributes(converted.new_spn.scope))
            #print(f'# {attrs}')

            if args.count_queries > 0:
                data: pd.DataFrame = read_table_csv(schema.tables[0], csv_seperator=';')
                attribute_ranges = { key: Domain(data[f'line_item_sanitized.{key}'], data.dtypes[f'line_item_sanitized.{key}'] == np.int64)
                                     for key in attribute_types.keys() }
                attributes = schema.tables[0].attributes[1:]

                for _ in range(100):
                    random.shuffle(attributes)
                    n = random.randint(1, 4)
                    attrs = attributes[0:n]
                    op = random.choice(['<=', '>='])
                    cond = ' AND '.join(f'{attr} {random.choice(["<=", ">="])} {attribute_ranges[attr].sample()}' for attr in attrs)
                    print(f'SELECT COUNT(*) FROM line_item_sanitized WHERE {cond};')


            if args.plot:
                plt.figure()
                n = len(converted.reduced_histograms)
                m = math.ceil(math.sqrt(n))

                for i, (var, reduced_histogram) in enumerate(converted.reduced_histograms.items()):
                    old_size = len(reduced_histogram.old_densities)
                    print(f'node {var} was reduced to {args.max_histogram_size} from old size {old_size}')

                    plt.subplot(m, m, i + 1)

                    # plot old
                    old_probs = reduced_histogram.old_densities
                    old_xs = list(range(len(old_probs)))
                    plt.bar(x=old_xs, height=old_probs)

                    # plot new
                    new_probs = reduced_histogram.new_densities
                    new_xs = list(range(len(new_probs)))
                    plt.bar(x=new_xs, height=new_probs)

                    print(f'old={sum(old_probs)} new={sum(new_probs)}')


                plt.show()

