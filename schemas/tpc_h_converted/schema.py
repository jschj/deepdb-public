from ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_tpc_h_converted_schema(csv_path):
    schema = SchemaGraph()
    schema.add_table(Table('line_item_sanitized',
                           attributes=['orderkey', 'partkey', 'suppkey', 'linenumber', 'quantity', 'extendedprices',
                                       'discount', 'tax', 'returnflag', 'linestatus', 'shipdate', 'commitdate',
                                       'receiptdate', 'shipinstruct', 'shipmode'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('line_item_sanitized_converted'),
                           table_size=6001216, primary_key=[],
                           sample_rate=6001216 / 6001216
                           ))

    return schema
