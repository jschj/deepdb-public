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

from xspn.domain import Domain
from xspn.schema import get_dataset_schema


def _populate_database(schema, attr_types, db_name, csv_path):
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


def _get_db_and_table_name(dataset, csv_path):
    schema, attr_types = get_dataset_schema(dataset, csv_path)

    if dataset == 'tpc-h':
        db_name = 'tpc_h'
        table_name = 'line_item_sanitized'
    elif dataset == 'tpc-h-converted':
        db_name = 'tpc_h_converted'
        table_name = 'line_item_sanitized'
    else:
        raise Exception(f'unknown dataset {dataset}')


    return schema, attr_types, db_name, table_name


def populate_database(dataset, csv_path):
    schema, attr_types, db_name, _ = _get_db_and_table_name(dataset, csv_path)
    _populate_database(schema=schema, attr_types=attr_types, db_name=db_name, csv_path=csv_path)


def generate_queries(dataset, csv_path, count=100):
    schema, attribute_types, _, table_name = _get_db_and_table_name(dataset, csv_path)

    for table in schema.tables:
        # remove 'index' column
        attributes = table.attributes[1:]
        table_name: str = table.table_name

        data: pd.DataFrame = read_table_csv(table, csv_seperator=';')

        # generate LEQ queries
        attribute_ranges = { key: Domain(data[f'{table_name}.{key}'], data.dtypes[f'{table_name}.{key}'] == np.int64)
                            for key in attribute_types.keys() }

        for _ in range(count):
            random.shuffle(attributes)
            n = random.randint(1, 4)
            attrs = attributes[0:n]
            op = random.choice(['<=', '>='])
            cond = ' AND '.join(f'{attr} {random.choice(["<="])} {attribute_ranges[attr].sample()}' for attr in attrs)
            print(f'SELECT COUNT(*) FROM {table_name} WHERE {cond};')
