python3 maqp.py --evaluate_cardinalities \
    --rdc_spn_selection \
    --max_variants 1 \
    --pairwise_rdc_path ../imdb-benchmark/spn_ensembles/pairwise_rdc.pkl \
    --dataset imdb-light \
    --target_path ./baselines/cardinality_estimation/results/deepDB/imdb_light_model_based_budget_5.csv \
    --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_relationships_imdb-light_1000000.pkl \
    --query_file_location ./benchmarks/job-light/sql/job_light_queries.sql \
    --ground_truth_file_location ./benchmarks/job-light/sql/job_light_true_cardinalities.csv
