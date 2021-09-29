import argparse
from datetime import datetime
import pickle
import itertools
import sys

from rspn_to_xspn import remap_data
from xspn.domain import remap_data_without_types

import pandas as pd
import numpy as np

from data_preparation.prepare_single_tables import read_table_csv

from schemas.tpc_h.schema import gen_tpc_h_schema
from xspn.compare_spns import compare_tpc_h, compare_spns
from xspn.schema import get_dataset_schema
from xspn.conversion import rspn_to_xspn_simple, xspn_to_str
from xspn.generate_queries import populate_database, generate_queries, compute_ground_truth
from xspn.expectation import rspn_expectation, spflow_expectation, original_expectation


def _convert_data(dataset: str, csv_path: str, out_file_path: str, max_domain_value: int):
    schema, attribute_types = get_dataset_schema(dataset, csv_path)
    df = read_table_csv(schema.tables[0], csv_seperator=';')
    remapped_df = remap_data(df, attribute_types, max_domain_value)
    remapped_df.to_csv(out_file_path, sep=';', header=False)


def _compare_spns(dataset: str, query_file_path: str, spn_paths: list[str], max_domain_values: list[int]):
    schema, attribute_types = get_dataset_schema(dataset, '../tpc-h/line_item_sanitized.csv')
    df = read_table_csv(schema.tables[0], csv_seperator=';')
    result = np.array(compare_spns(query_file_path, df, [path for path in spn_paths], [int(val) for val in max_domain_values], schema, attribute_types))
    result *= 6001215

    for res, spn_path, max_domain_value in zip(result, spn_paths, max_domain_values):
        print(f'{spn_path} with max domain value {max_domain_value} results in:')
        print(list((result[0] - res) / result[0]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--convert', help='Convert the domain of some data to a domain of a lower size. --domain_size, --in_file and --out_file are required.', action='store_true')
    parser.add_argument('--compare', help='', action='store_true')

    parser.add_argument('--dataset', help='code name of the dataset which is processed', type=str)
    
    parser.add_argument('--csv_path', help='', type=str)
    parser.add_argument('--out_file', type=str)

    parser.add_argument('--max_domain_values', nargs='*', help='the new domain [0, max_domain_value] to which the old domain is mapped')
    parser.add_argument('--spns', nargs='*')
    parser.add_argument('--query_file', type=str)

    parser.add_argument('--pkl_path', type=str, help='')
    parser.add_argument('--xspn', help='', action='store_true')

    parser.add_argument('--gen', help='generates LEQ count queries for random attributes with random values and prints them out', action='store_true')
    parser.add_argument('--count', help='the number of queries to be generated', type=int, default=100)
    parser.add_argument('--as_csv', help='should the queries be generated in CSV format', action='store_true')
    parser.add_argument('--seed', help='set the seed for the randomly generated queries', type=int, default=123456)

    parser.add_argument('--populate', help='populates the database with data from the csv file', action='store_true')

    parser.add_argument('--ground_truth', help='Computes the correct values for the given queries by QUERY_FILE. Requires DATASET, PKL_PATH and QUERY_FILE.', action='store_true')

    args = parser.parse_args()

    if args.convert:
        _convert_data(args.dataset, args.csv_path, args.out_file, args.max_domain_value)
    elif args.compare:
        _compare_spns(args.dataset, args.query_file, args.spns, args.max_domain_values)
    elif args.xspn:
        schema, attribute_types = get_dataset_schema(args.dataset, args.csv_path)

        with open(args.pkl_path, 'rb') as f:
            pkl = pickle.load(f)

            for i, spn in enumerate(pkl.spns):
                rspn = spn.mspn
                xspn = rspn_to_xspn_simple(rspn)
                # NOTE: Experimental!
                attributes = list(itertools.chain(*[table.attributes for table in schema.tables]))
                xspn_str = xspn_to_str(xspn, attributes)

                print(xspn_str)
    elif args.gen:
        generate_queries(args.dataset, args.csv_path, args.count, args.as_csv, args.seed)
    elif args.populate:
        populate_database(args.dataset, args.csv_path)
    elif args.ground_truth:
        if args.as_csv:
            evidence = np.genfromtxt(args.query_file, delimiter=';')
            # replace -1 with nan
            evidence = np.where(evidence == -1, np.nan, evidence)
            #evidence = np.array([evidence[2]])
        else:
            raise NotImplementedError('non CSV formats are currently not supported')

        schema, attribute_types = get_dataset_schema(args.dataset, args.csv_path)

        with open(args.pkl_path, 'rb') as f:
            pkl = pickle.load(f)

            # TODO: output is bugged for multiple SPNs
            for i, spn in enumerate(pkl.spns):
                rspn = spn.mspn
                xspn = rspn_to_xspn_simple(rspn)
                result = rspn_expectation(xspn, evidence) * 6001215
                #other_result = spflow_expectation(xspn, evidence) * 6001215
                other_result = original_expectation(rspn, evidence) * 6001215
                #np.savetxt(sys.stdout.buffer, result, delimiter=';')
                
                other_result = other_result.reshape(-1)
                print(result)
                print(other_result)
                delta = other_result - result
                print(delta)
