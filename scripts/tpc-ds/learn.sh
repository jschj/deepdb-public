#python3 maqp.py --generate_hdf \
#    --dataset imdb-light \
#    --csv_seperator , \
#    --csv_path ../imdb-benchmark \
#    --hdf_path ../imdb-benchmark/gen_single_light \
#    --max_rows_per_hdf_file 100000000
#
python3 maqp.py --generate_sampled_hdfs \
    --dataset imdb-light \
    --hdf_path ../imdb-benchmark/gen_single_light \
    --max_rows_per_hdf_file 100000000 \
    --hdf_sample_size 10000

python3 maqp.py --generate_ensemble \
    --dataset imdb-light  \
    --samples_per_spn 10000000 10000000 1000000 1000000 1000000 \
    --ensemble_strategy rdc_based \
    --hdf_path ../imdb-benchmark/gen_single_light \
    --max_rows_per_hdf_file 100000000 \
    --samples_rdc_ensemble_tests 10000 \
    --ensemble_path ../imdb-benchmark/spn_ensembles \
    --database_name imdb \
    --post_sampling_factor 10 10 5 1 1 \
    --ensemble_budget_factor 5 \
    --ensemble_max_no_joins 3 \
    --pairwise_rdc_path ../imdb-benchmark/spn_ensembles/pairwise_rdc.pkl
