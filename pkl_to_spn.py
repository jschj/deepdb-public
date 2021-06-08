from ensemble_compilation.graph_representation import Table
import pickle
import sys

from spn.io.Text import spn_to_str_ref_graph
from spn.io.Graphics import plot_spn, plot_spn2
from spn.structure.leaves.histogram.Histograms import Histogram

from rspn.structure.base import Sum
from rspn.structure.leaves import Categorical, IdentityNumericLeaf
from schemas.tpc_h.schema import gen_tpc_h_schema
from data_preparation.prepare_single_tables import read_table_csv

import matplotlib.pyplot as plt
import pandas as pd

# the first attribute seems to be dropped
ATTRIBUTES = ['orderkey', 'partkey', 'suppkey', 'linenumber', 'quantity', 'extendedprices',
              'discount', 'tax', 'returnflag', 'linestatus', 'shipdate', 'commitdate',
              'receiptdate', 'shipinstruct', 'shipmode']


def scope_to_attributes(scope):
    return [ATTRIBUTES[s] for s in scope]


def sum_to_str(node, feature_names, node_to_str):
    w = node.weights
    ch = node.children
    sumw = ", ".join(map(lambda i: "%s*%s" % (w[i], ch[i].name), range(len(ch))))
    child_str = "".join(map(lambda c: spn_to_str_ref_graph(c, feature_names, node_to_str), node.children))
    child_str = child_str.replace("\n", "\n\t")
    return "%s SumNode(%s){\n\t%s}\n" % (str(node), sumw, child_str)


def categorical_to_str(node, feature_names, node_to_str):
    probs = '...' #', '.join(str(p) for p in node.p)
    scope = ', '.join(scope_to_attributes(node.scope))
    cardinality = node.cardinality
    n = len(node.p)
    return f'{str(node)} Categorical([{scope}], size={n})\n'


def identity_numeric_leaf_to_str(node, feature_names, node_to_str):
    #unique_vals, probabilities, scope

    vals = '...' #', '.join(str(v) for v in node.unique_vals)
    probs = '...' #', '.join(str(p) for p in node.prob_sum)
    scope = ', '.join(scope_to_attributes(node.scope))
    size = len(node.unique_vals)

    return f'{str(node)} IdentityNumericLeaf([{scope}], size={size})\n'


node_to_str = {
    Sum: sum_to_str,
    Categorical: categorical_to_str,
    IdentityNumericLeaf: identity_numeric_leaf_to_str
}

# make things ready for xspn

def categorical_to_histogram(node):
    raise NotImplementedError
    return Histogram(breaks, densities, bin_repr_points, scope)


def identity_numeric_to_histogram(node):
    raise NotImplementedError


def plot_distributions(node):
    if isinstance(node, IdentityNumericLeaf):
        size = len(node.unique_vals)

        if size >= 100:
            plt.figure()
            xs = node.unique_vals
            ys = node.return_histogram()
            #print(node.mean)
            plt.scatter(xs, ys, s=0.1)
            plt.show()

            scope = node.scope
            print(f'IdentityNumericLeaf size={size} num-scope={scope} scope={scope_to_attributes(scope)}')
    elif isinstance(node, Categorical):
        size = len(node.p)

        if size >= 100:
            #plt.figure()
            #xs = range(size)
            #ys = node.p
            #plt.scatter(xs, ys, s=0.2)
            #plt.show()
            scope = node.scope
            print(f'Categorical size={size} num-scope={scope} scope={scope_to_attributes(scope)}')
    else:
        # Sum or Product
        for child in node.children:
            plot_distributions(child)


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print(f'Usage: python3 {__file__} <pickle file path> <data csv path>')
        sys.exit()

    pkl_path = sys.argv[1]
    csv_path = sys.argv[2]

    schema = gen_tpc_h_schema(csv_path)
    table: Table = schema.tables[0]

    table_data: pd.DataFrame = read_table_csv(table, csv_seperator=';')

    for col in table_data.columns:
        unique = pd.unique(table_data[col])
        #print(f'{col}: {len(unique)}')

    #sys.exit()

    with open(pkl_path, 'rb') as f:
        pkl = pickle.load(f)

        for i, spn in enumerate(pkl.spns):
            mspn = spn.mspn
            plot_distributions(mspn)

            #print(spn_to_str_ref_graph(mspn, node_to_str=node_to_str))



            #plot_spn2(spn.mspn, f'plot_{i}.pdf')
