from ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_tpc_h_schema(csv_path):
    schema = SchemaGraph()
    schema.add_table(Table('lineitem',
                           attributes=['orderkey', 'partkey', 'suppkey', 'linenumber', 'quantity', 'extendedprices',
                                       'discount', 'tax', 'returnflag', 'linestatus', 'shipdate', 'commitdate',
                                       'receiptdate', 'shipinstruct', 'shipmode'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('lineitem'),
                           table_size=6001216, primary_key=[],
                           sample_rate=6001216 / 6001216,
                           no_compression=['orderkey', 'partkey', 'suppkey', 'linenumber', 'quantity', 'extendedprices',
                                           'discount', 'tax', 'returnflag', 'linestatus', 'shipdate', 'commitdate',
                                           'receiptdate', 'shipinstruct', 'shipmode']
                           ))

    return schema
