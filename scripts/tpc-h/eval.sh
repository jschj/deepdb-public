python3 maqp.py --evaluate_cardinalities \
    --max_variants 1 \
    --dataset tpc-h \
    --target_path ../tpc-h/cardinality_estimation/tpc-h.csv \
    --ensemble_location ../tpc-h/spn_ensembles/ensemble_single_tpc-h_10000000.pkl \
    --query_file_location ../tpc-h/queries/count_queries.sql \
    --ground_truth_file_location ../tpc-h/queries/ground_truth_count.pkl \
    --ensemble_strategy single
