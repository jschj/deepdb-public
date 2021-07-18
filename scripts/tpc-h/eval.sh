python3 maqp.py --evaluate_cardinalities \
    --max_variants 1 \
    --dataset tpc-h \
    --target_path ../tpc-h/cardinality_estimation/count_queries_leq.csv \
    --ensemble_location ../tpc-h/spn_ensembles/ensemble_single_tpc-h_10000000.pkl \
    --query_file_location ../tpc-h/queries/count_queries_leq.sql \
    --ground_truth_file_location ../tpc-h/queries/count_queries_leq_truth.pkl \
    --ensemble_strategy single
