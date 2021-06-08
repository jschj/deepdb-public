python3 maqp.py --cardinalities_ground_truth \
    --dataset tpc-h \
    --query_file_location ./benchmarks/tpc_ds_single_table/sql/aqp_queries.sql \
    --target_path ./benchmarks/tpc_ds_single_table/ground_truth_1t.pkl \
    --database_name tcpds
