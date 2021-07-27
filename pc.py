from cdt.data import load_dataset
import networkx as nx
from utils.pc_utils import pcalg, resit
import numpy as np
import matplotlib.pyplot as plt
import time 
import os 
import matplotlib.pyplot as plt
import networkx as nx
from eval.plots import plot_graph
import cdt
import netrd
import pandas as pd
from utils import set_seeds


def pc_estimator(input_dataset: pd.DataFrame, target_graph: nx.Graph, indep_test,
                 device, kwargs_m=None, kwargs_c=None):
    """
    Estimate the PC algorithm. 
    :param input_dataset: the dataset to be used for estimation.
    :param target_graph: the target graph.
    :param indep_test: the function to be used for estimation.
    :return: the estimated graph.
    """
    # Estimate the skeleton graph
    pc = pcalg(dataset=input_dataset.to_numpy(), feature_names=input_dataset.columns.to_list(),
                kwargs_m=kwargs_m, kwargs_c=kwargs_c, device=device)
    pc.identify_skeleton_original(indep_test=indep_test)

    # Orient the graph
    pc.orient_graph(indep_test=indep_test, alpha=0.05)

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
    distance_measure = netrd.distance.Hamming()
    shd = distance_measure.dist(G1=target_graph, G2=estimated_graph)
    print('Structural hamming distance: {}'.format(shd))
    return shd
