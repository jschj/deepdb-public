python3 maqp.py --generate_ensemble \
    --dataset tpc-h \
    --samples_per_spn 1000000 \
    --ensemble_strategy single \
    --hdf_path ../tpc-h-$1/gen_single_light \
    --ensemble_path ../tpc-h-$1/spn_ensembles \
    --rdc_threshold 0.3 \
    --post_sampling_factor 10
