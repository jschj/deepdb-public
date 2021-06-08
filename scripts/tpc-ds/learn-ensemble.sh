python3 maqp.py --generate_ensemble \
    --dataset imdb-light \
    --samples_per_spn 1000000 1000000 1000000 1000000 1000000 \
    --ensemble_strategy relationship \
    --hdf_path ../imdb-benchmark/gen_single_light \
    --ensemble_path ../imdb-benchmark/spn_ensembles \
    --max_rows_per_hdf_file 100000000 \
    --post_sampling_factor 10 10 5 1 1
