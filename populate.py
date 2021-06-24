from ensemble_compilation.graph_representation import Table
from schemas.tpc_h.schema import gen_tpc_h_schema
from data_preparation.prepare_single_tables import read_table_csv

import pickle
import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    dataset = sys.argv[1]
