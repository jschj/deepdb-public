python3 maqp.py --generate_hdf \
    --dataset tpc-h \
    --csv_seperator ';' \
    --csv_path ../tpc-h-$1 \
    --hdf_path ../tpc-h-$1/gen_single_light \
    --max_rows_per_hdf_file 100000000

