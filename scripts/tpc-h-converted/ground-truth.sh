python3 maqp.py --cardinalities_ground_truth \
    --dataset tpc-h-converted \
    --query_file_location ../tpc-h-converted/queries/count_queries_leq.sql \
    --target_path ../tpc-h-converted/queries/ground_truth_count_leq.pkl \
    --database_name tpc_h_converted
