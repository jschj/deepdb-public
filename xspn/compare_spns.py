import math
import argparse
import pickle
from typing import Tuple
import pandas as pd

from schemas.tpc_h_converted.schema import gen_tpc_h_converted_schema
from schemas.tpc_h.schema import gen_tpc_h_schema
from data_preparation.prepare_single_tables import read_table_csv

from cardinality_custom import expectation
from evaluation.utils import parse_query
from ensemble_compilation.graph_representation import Query, QueryType, SchemaGraph, Table

from xspn.domain import generate_domain_mapping


def _rspn_expectation(spn, query, schema: SchemaGraph):
    return expectation(spn, schema.tables[0], query, schema)


def _build_query_str(leq_query: dict, table_name: str) -> str:
    cond_str = ' AND '.join(f'{attr} <= {val}' for attr, val in leq_query.items())
    return f'SELECT COUNT(*) FROM {table_name} WHERE {cond_str}'


def _compare_query(original_spn, original_query: dict, converted_spn, domain_mapping: dict, schema: SchemaGraph):
    converted_query = {k: domain_mapping[k].convert(v) for k, v in original_query.items()}

    original_query_str = _build_query_str(original_query, schema.tables[0].table_name)
    converted_query_str = _build_query_str(converted_query, schema.tables[0].table_name)

    print(f'original query: {original_query_str}')
    print(f'converted query: {converted_query_str}')

    original_exp = _rspn_expectation(original_spn, original_query_str, schema)
    converted_exp = _rspn_expectation(converted_spn, converted_query_str, schema)

    return original_exp, converted_exp


def _benchmark_spn(spn, queries: list[dict], domain_mapping: dict, schema: SchemaGraph):
    def benchmark_single_query(query):
        converted_query = {k: domain_mapping[k].convert(v) for k, v in query.items()}
        converted_query_str = _build_query_str(converted_query, schema.tables[0].table_name)
        converted_exp = _rspn_expectation(spn, converted_query_str, schema)

        return converted_exp


    return [benchmark_single_query(query) for query in queries]


def _benchmark_spns(spns, queries: list[dict], df: pd.DataFrame, max_domain_values: list[int], attribute_types: dict, schema: SchemaGraph):
    results = []
    
    for spn, max_domain_value in zip(spns, max_domain_values):
        domain_mapping = generate_domain_mapping(df, attribute_types, max_domain_value + 1)
        results.append(_benchmark_spn(spn, queries, domain_mapping, schema))


    return results


def other_compare_spns(original_spn, converted_spn, schema: SchemaGraph, sql_query_file_path, domain_mapping: dict):
    def split_cond_str(cond_str: str):
        parts = cond_str.split('<=')
        return parts[0], float(parts[1])


    with open(sql_query_file_path, 'r') as query_file:
        for line in query_file.readlines():
            query = parse_query(line, schema)
            original_query = dict(split_cond_str(cond) for (_, cond) in query.conditions)

            org, conv = _compare_query(original_spn, original_query, converted_spn, domain_mapping, schema)
            err = conv - org
            rel_err = err / org if org != 0 else 0
            print(f'original={org * 6001215} converted={conv * 6001215} error={err} relative error={rel_err}')


def compare_tpc_h():
    #parser = argparse.ArgumentParser()
    #args = parser.parse_args()

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

    tpc_h_schema = gen_tpc_h_schema('../tpc-h/line_item_sanitized.csv')
    tpc_h_converted_schema = gen_tpc_h_converted_schema('../tpc-h-converted/line_item_sanitized_converted.csv')

    original_path = '../tpc-h/spn_ensembles/ensemble_single_tpc-h_10000000.pkl'
    converted_path = '../tpc-h-converted/spn_ensembles/ensemble_single_tpc-h-converted_1000000.pkl'

    original_df = read_table_csv(tpc_h_schema.tables[0], csv_seperator=';')
    converted_df = read_table_csv(tpc_h_converted_schema.tables[0], csv_seperator=';')

    domain_mapping = generate_domain_mapping(original_df, attribute_types)

    with open(original_path, 'rb') as original, open(converted_path, 'rb') as converted:
        org_pkl = pickle.load(original)
        conv_pkl = pickle.load(converted)

        for i, (org_spn, conv_spn) in enumerate(zip(org_pkl.spns, conv_pkl.spns)):
            other_compare_spns(org_spn.mspn, conv_spn.mspn, tpc_h_schema, '../tpc-h/queries/count_queries_leq.sql', domain_mapping)


def compare_spns(query_file_path: str, df: pd.DataFrame, spn_paths: list[str], max_domain_values: list[int], schema: SchemaGraph, attribute_types: dict):
    def open_spn(path):
        with open(path, 'rb') as f:
            return pickle.load(f).spns[0].mspn


    def split_cond_str(cond_str: str):
        parts = cond_str.split('<=')
        return parts[0], float(parts[1])


    spns = [open_spn(path) for path in spn_paths]

    with open(query_file_path, 'r') as f:
        parsed_queries = [parse_query(query, schema) for query in f.readlines()]
        query_dicts = [dict(split_cond_str(cond) for (_, cond) in query.conditions)
                       for query in parsed_queries]


    results = _benchmark_spns(spns, query_dicts, df, max_domain_values, attribute_types, schema)

    return results
