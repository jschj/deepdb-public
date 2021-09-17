import sys
import random
from copy import copy
import argparse
import psycopg2

import pandas as pd
from pandas.core.algorithms import isin
from pandas.core.frame import DataFrame
import numpy as np

from schemas.tpc_h.schema import gen_tpc_h_schema
from schemas.tpc_h_converted.schema import gen_tpc_h_converted_schema
from ensemble_compilation.graph_representation import Table
from data_preparation.prepare_single_tables import read_table_csv


def populate_database(schema, attr_types, db_name, csv_path):
    connection = psycopg2.connect(user='postgres',
                                  password='postgres',
                                  host='localhost',
                                  port='5432',
                                  database=db_name)

    def exec_sql(sql_cmd):
        print(f'Executing {sql_cmd}')

        try:
            cursor = connection.cursor()
            cursor.execute(sql_cmd)
            connection.commit()
        except Exception as e:
            print(e)


        print('Done')


    for table in schema.tables:
        #print(f'Loading CSV file for {table.table_name}...')
        #table_data: pd.DataFrame = read_table_csv(table, csv_seperator=';')
        #print('Done')

        print('Dropping (old) table...')
        exec_sql(f'DROP TABLE IF EXISTS {table.table_name}')

        print('Creating table...')
        attributes = ', '.join(f'{attr} {attr_types[attr]}' for attr in table.attributes)
        exec_sql(f'CREATE TABLE {table.table_name} ({attributes})')

        print(f'Inserting data...')
        with open(csv_path, 'r') as file:
            cursor = connection.cursor()
            cursor.copy_from(file, table.table_name, sep=';')
            connection.commit()

        print('Done')


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
    parser.add_argument('--populate_db', help='Load dataset into postgres database', action='store_true')

    args = parser.parse_args()

    if args.dataset == 'tpc-h':
        schema = gen_tpc_h_schema(args.csv_path)
        table_name = 'line_item_sanitized'
        db_name = 'tpc_h'
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
    elif args.dataset == 'tpc-h-converted':
        schema = gen_tpc_h_converted_schema(args.csv_path)
        table_name = 'line_item_sanitized'
        db_name = 'tpc_h_converted'
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
        print(f'ERROR: Unknown dataset {args.dataset}')
        sys.exit(1)

    attributes = schema.tables[0].attributes[1:]
    data: pd.DataFrame = read_table_csv(schema.tables[0], csv_seperator=';')
    #table_name: str = schema.tables[0].csv_file_location.split('/')[-1].split('.')[0]
    
    if args.populate_db:
        populate_database(schema, attribute_types, db_name, args.csv_path)


    


    # generate LEQ queries
    attribute_ranges = { key: Domain(data[f'{table_name}.{key}'], data.dtypes[f'{table_name}.{key}'] == np.int64)
                         for key in attribute_types.keys() }

    for _ in range(100):
        random.shuffle(attributes)
        n = random.randint(1, 4)
        attrs = attributes[0:n]
        op = random.choice(['<=', '>='])
        cond = ' AND '.join(f'{attr} {random.choice(["<="])} {attribute_ranges[attr].sample()}' for attr in attrs)
        print(f'SELECT COUNT(*) FROM {table_name} WHERE {cond};')
