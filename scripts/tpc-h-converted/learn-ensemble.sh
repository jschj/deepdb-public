python3 maqp.py --generate_ensemble \
    --dataset tpc-h-converted \
    --samples_per_spn 1000000 \
    --ensemble_strategy single \
    --hdf_path ../tpc-h-converted/gen_single_light \
    --ensemble_path ../tpc-h-converted/spn_ensembles \
    --rdc_threshold 0.3 \
    --post_sampling_factor 10