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
from ensemble_compilation.physical_db import DBConnection

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import psycopg2

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

        if size > 256:
            plt.figure()
            
            probs = node.return_histogram()
            hist, bin_edges = np.histogram(a=probs, bins=256, density=True)
            hist = hist / np.sum(hist)
            print(f'hist sum = {np.sum(hist)}')
            #print(f'hist={hist} bin_edges={bin_edges}')
            plt.hist(hist, bins=256)

            xs = node.unique_vals
            ys = node.return_histogram()
            plt.scatter(xs, ys, s=0.1)

            plt.show()

            scope = node.scope
            print(f'IdentityNumericLeaf size={size} num-scope={scope} scope={scope_to_attributes(scope)}')
    elif isinstance(node, Categorical):
        size = len(node.p)

        if size > 256:
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='abc', help='Which dataset to be used')
    parser.add_argument('--pickle_path', default='', help='Pickle file path')
    parser.add_argument('--csv_path', default='', help='CSV file path')
    parser.add_argument('--populate_postgres', help='Load dataset into postgres database')

    args = parser.parse_args()

    print(f'dataset={args.dataset}')

    if args.dataset == 'tpc-h':
        schema = gen_tpc_h_schema(args.csv_path)
    else:
        raise ValueError('Dataset unknown')

    connection = psycopg2.connect(user='postgres',
                                  password='postgres',
                                  host='localhost',
                                  port='5432',
                                  database='tpc_h')

    for table in schema.tables:
        #print(f'Loading CSV file for {table.table_name}...')
        #table_data: pd.DataFrame = read_table_csv(table, csv_seperator=';')
        #print('Done')

        print('Creating table...')

        # 1;1;67310;7311;2;36.0;45983.16;0.09;0.06;2;0;829260000;825462000;829951200;3;4

        attr_types = {
            'id': 'bigint',
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

        attributes = ', '.join(f'{attr} {attr_types[attr]}' for attr in table.attributes)
        sql_cmd = f'CREATE TABLE {table.table_name} ({attributes})'

        print(f'Executing {sql_cmd}')
        try:
            cursor = connection.cursor()
            cursor.execute(sql_cmd)
            connection.commit()
        except Exception as e:
            print(e)
        print('Done')

        print(f'Inserting data...')

        with open(args.csv_path, 'r') as file:
            cursor = connection.cursor()
            cursor.copy_from(file, table.table_name, sep=';')
            connection.commit()

        print('Done')


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
