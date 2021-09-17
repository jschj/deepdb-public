from schemas.tpc_h.schema import gen_tpc_h_schema
from schemas.tpc_h_converted.schema import gen_tpc_h_converted_schema


def get_dataset_schema(dataset: str, csv_path: str):
    if dataset == 'tpc-h':
        schema = gen_tpc_h_schema(csv_path)
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
        raise ValueError(f'unknown dataset {dataset}')

    
    return schema, attribute_types
