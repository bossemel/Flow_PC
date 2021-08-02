import networkx as nx
from utils.pc_utils import pcalg
import networkx as nx
import cdt
import pandas as pd


def pc_estimator(input_dataset: pd.DataFrame, indep_test, alpha,
                 device, kwargs_m=None, kwargs_c=None):
    """
    Estimate the PC algorithm. 
    :param input_dataset: the dataset to be used for estimation.
    :param indep_test: the function to be used for estimation.
    :return: the estimated graph.
    """
    # Estimate the skeleton graph
    pc = pcalg(dataset=input_dataset.to_numpy(), feature_names=input_dataset.columns.to_list(),
                kwargs_m=kwargs_m, kwargs_c=kwargs_c, device=device)
    pc.identify_skeleton_original(indep_test=indep_test, alpha=alpha)

    # # Orient the graph
    # pc.orient_graph(indep_test=indep_test, alpha=alpha)
    # Instead, create bidirectional edges:
    pc.G = pc.G.to_directed()

    # Return the graph with labels
    estimated_graph = pc.G
    estimated_graph = nx.relabel.relabel_nodes(estimated_graph, pc.features)
    return estimated_graph


def shd_calculator(target_graph: nx.Graph, estimated_graph: nx.Graph):
    """
    Calculate the structural hamming distance between two graphs.
    :param target_graph: the target graph.
    :param estimated_graph: the estimated graph.
    :return: the structural hamming distance.
    """
    shd = cdt.metrics.SHD(target_graph, estimated_graph)
    print('Structural hamming distance: {}'.format(shd))
    return shd
