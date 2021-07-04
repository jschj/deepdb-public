from pkl_to_spn import scope_to_attributes
from ensemble_compilation.graph_representation import Table
import pickle
import math

from spn.io.Text import spn_to_str_ref_graph
from spn.structure.leaves.histogram.Histograms import Histogram

from rspn.structure.base import Sum
from rspn.structure.leaves import IdentityNumericLeaf
from schemas.tpc_h.schema import gen_tpc_h_schema

import argparse


scope_to_attributes = lambda x: ''


def sum_to_str(node, feature_names, node_to_str):
    w = node.weights
    ch = node.children
    sumw = ", ".join(map(lambda i: "%s*%s" % (w[i], ch[i].name), range(len(ch))))
    child_str = "".join(map(lambda c: spn_to_str_ref_graph(c, feature_names, node_to_str), node.children))
    child_str = child_str.replace("\n", "\n\t")
    return "%s SumNode(%s){\n\t%s}\n" % (str(node), sumw, child_str)


def identity_numeric_leaf_to_str(node, feature_names, node_to_str):
    #unique_vals, probabilities, scope

    vals = '...' #', '.join(str(v) for v in node.unique_vals)
    probs = '...' #', '.join(str(p) for p in node.prob_sum)
    scope = ', '.join(scope_to_attributes(node.scope))
    size = len(node.unique_vals)

    return f'{str(node)} IdentityNumericLeaf([{scope}], size={size})\n'


def histogram_to_str(node, feature_names, node_to_str):
    scope = ','.join(scope_to_attributes(node.scope))
    breaks = ','.join(f'{b}.' for b in node.breaks)
    probabilities = ','.join(str(d) for d in node.densities)
    return f'{str(node)} Histogram({scope}|[{breaks}];[{probabilities}])\n'


node_to_str = {
    Sum: sum_to_str,
    IdentityNumericLeaf: identity_numeric_leaf_to_str,
    Histogram: histogram_to_str
}


def reshape_histogram(alphas, m):
    """
    This function takes a histogram and puts together a condensed histogram of given size.
    """
    betas = []
    n = len(alphas)

    for i in range(m):
        li = n / m * i
        hi = n / m * (i + 1)

        complete_alphas = sum(alphas[math.ceil(li) : math.floor(hi) + 1])
        front_alphas = (math.ceil(li) - li) * alphas[math.floor(li)]
        
        if hi < n:
            back_alphas = (hi - math.floor(hi)) * alphas[math.floor(hi)]
        else:
            back_alphas = 0

        betas.append(complete_alphas + front_alphas + back_alphas)

    return betas


class ConvertedSPN:
    def __init__(self, spn, table, max_histogram_size) -> None:
        self.old_spn = spn
        self.reduced_histograms = dict()
        self.new_spn = self._convert_spn(spn, table, max_histogram_size)

    def _convert_spn(self, node, table: Table, max_histogram_size):
        if isinstance(node, IdentityNumericLeaf):
            size = len(node.unique_vals)
            points = [] # TODO: What's this?

            if size > max_histogram_size:
                densities = reshape_histogram(node.return_histogram(), max_histogram_size)
                breaks = list(range(max_histogram_size + 1))
            else:
                breaks = list(range(size + 1))
                densities = node.return_histogram(copy=True)

            histogram = Histogram(breaks=breaks, densities=densities, bin_repr_points=points, scope=node.scope)
            histogram.id = node.id

            if size > max_histogram_size:
                self.reduced_histograms[node.id] = (size, histogram)

            return histogram
        else:
            for i in range(len(node.children)):
                node.children[i] = self._convert_spn(node.children[i], table, max_histogram_size)

            return node


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='abc', help='Which dataset to be used')
    parser.add_argument('--pickle_path', default='', help='Pickle file path')
    parser.add_argument('--csv_path', default='', help='CSV file path')
    parser.add_argument('--max_histogram_size', default=256)

    args = parser.parse_args()

    if args.dataset == 'tpc-h':
        schema = gen_tpc_h_schema(args.csv_path)
    else:
        raise ValueError('Dataset unknown')

    if len(schema.tables) != 1:
        raise NotImplementedError('Conversion of RSPNs for more than 1 table is currently not supported!')

    with open(args.pickle_path, 'rb') as f:
        pkl = pickle.load(f)

        for i, spn in enumerate(pkl.spns):
            scope_to_attributes = lambda scope: [schema.tables[0].attributes[s] for s in scope]

            mspn = spn.mspn
            converted = ConvertedSPN(mspn, schema.tables[0], args.max_histogram_size)
            #sys.exit()

            #print(spn_to_str_ref_graph(converted.new_spn, node_to_str=node_to_str))
            #attrs = ';'.join(s for s in scope_to_attributes(converted.new_spn.scope))
            #print(f'# {attrs}')

            print(converted.reduced_histograms.keys())
