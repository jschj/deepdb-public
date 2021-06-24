import sys
import random
from copy import copy
import argparse

import pandas as pd
from pandas.core.algorithms import isin
from pandas.core.frame import DataFrame
import numpy as np

from schemas.tpc_h.schema import gen_tpc_h_schema
from ensemble_compilation.graph_representation import Table
from data_preparation.prepare_single_tables import read_table_csv


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
    #parser.add_argument('--pickle_path', default='', help='Pickle file path')
    parser.add_argument('--csv_path', default='', help='CSV file path')
    #parser.add_argument('--populate_postgres', help='Load dataset into postgres database')

    args = parser.parse_args()

    if args.dataset == 'tpc-h':
        schema = gen_tpc_h_schema(args.csv_path)
        attribute_types = {
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
    else:
        print(f'ERROR: Unknown dataset {args.dataset}')
        sys.exit(1)

    attributes = schema.tables[0].attributes[1:]
    data: pd.DataFrame = read_table_csv(schema.tables[0], csv_seperator=';')
    
    #print(data)
    #print(min(data['line_item_sanitized.partkey']))
    #print(max(data['line_item_sanitized.partkey']))
    #print(data.dtypes)
    #print(data.dtypes['line_item_sanitized.partkey'])
    #print(data.dtypes['line_item_sanitized.partkey'] == np.int64)

    attribute_ranges = { key: Domain(data[f'line_item_sanitized.{key}'], data.dtypes[f'line_item_sanitized.{key}'] == np.int64)
                         for key in attribute_types.keys() }

    #for k, v in attribute_ranges.items():
    #    print(f'{k}: {v.min} {v.max} {v.integral}')

    for _ in range(100):
        random.shuffle(attributes)
        n = random.randint(1, 4)
        attrs = attributes[0:n]
        op = random.choice(['<=', '>='])
        cond = ' AND '.join(f'{attr} {random.choice(["<=", ">="])} {attribute_ranges[attr].sample()}' for attr in attrs)
        print(f'SELECT COUNT(*) FROM line_item_sanitized WHERE {cond};')
