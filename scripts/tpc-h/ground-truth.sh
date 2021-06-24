python3 maqp.py --cardinalities_ground_truth \
    --dataset tpc-h \
    --query_file_location ../tpc-h/queries/count_queries.sql \
    --target_path ../tpc-h/queries/ground_truth_count.pkl \
    --database_name tpc_h
