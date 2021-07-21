from cdt.data import load_dataset
import networkx as nx
from utils.pc_utils import pcalg, resit
from pgmpy.estimators import PC 
import numpy as np
import matplotlib.pyplot as plt
import time 


if __name__ == '__main__':
    s_data, s_graph = load_dataset('sachs')

    print(s_data.shape)

    pc = pcalg(dataset=s_data.to_numpy()[:, :10]) #, feature_names=features)
    pc.identify_skeleton_original(indep_test=resit)
    # pc_fit = PC(data=s_data_sample)
    # estimated_graph = pc_fit.build_skeleton(variant='parallel')
    # print(estimated_graph.number_of_edges())
    # print(estimated_graph.number_of_nodes())
    # print(estimated_graph.edges())
    # print(estimated_graph.nodes())

    # pcalg.estimate_skeleton(indep_test_func=ci_test_dis,
    #                                             data_matrix=s_data,
    #                                             alpha=0.05)

    # @Todo: use this one https://pgmpy.org/_modules/pgmpy/estimators/PC.html
    # @Todo: rewrite to allow these inputs: https://pgmpy.org/citests.html
    #print(s_data.shape)
    #print(type(s_graph), len(s_graph))
    #nx.draw(estimated_graph)
    pc.render_graph()
    plt.savefig('results/testing/est_graph.png')

    