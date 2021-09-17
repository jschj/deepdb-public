python3 maqp.py --evaluate_cardinalities \
    --max_variants 1 \
    --dataset tpc-h-converted \
    --target_path ../tpc-h-converted/cardinality_estimation/count_queries_leq.csv \
    --ensemble_location ../tpc-h-converted/spn_ensembles/ensemble_single_tpc-h-converted_1000000.pkl \
    --query_file_location ../tpc-h-converted/queries/count_queries_leq.sql \
    --ground_truth_file_location ../tpc-h-converted/queries/ground_truth_count_leq.pkl \
    --ensemble_strategy single
