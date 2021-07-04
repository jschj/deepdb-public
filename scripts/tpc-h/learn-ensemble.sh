python3 maqp.py --generate_ensemble \
    --dataset tpc-h \
    --samples_per_spn 1000 \
    --ensemble_strategy single \
    --hdf_path ../tpc-h/gen_single_light \
    --ensemble_path ../tpc-h/spn_ensembles \
    --rdc_threshold 0.3 \
    --post_sampling_factor 10