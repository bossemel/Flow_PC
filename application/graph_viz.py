import networkx as nx
from utils import create_folders
from options import TrainOptions
import os
from eval.plots import plot_graph


def recipr_graph():
    """
    Rename the nodes of the reciprocity graph.
    """
    # Create new folders
    args = TrainOptions().parse()
    args.exp_name = 'ebay_pc_recipr'
    args.flow_name = 'cop_flow'
    args.alpha_indep = 0.05
    add_name = 'flow'

    # Create Folders
    args = create_folders(args)

    # Read pickled graph
    undirected_graph = nx.read_gpickle(os.path.join(os.path.join(args.figures_path, add_name + 'est_graph.pickle')))
    print(nx.info(undirected_graph))

    # Show the nodes
    print(undirected_graph.nodes())
    print(undirected_graph.edges())
    
    # Rename nodes
    mapping = {'log_concessions': 'C1', 
               'log_opp_concessions': 'C2', 
               'log_offr_price': 'P', 
               'log_time_since_offer': 'T', 
               'log_hist': 'H1', 
               'log_opp_hist': 'H2'}
    undirected_graph = nx.relabel_nodes(undirected_graph, mapping)

    # Show the nodes
    print(undirected_graph.nodes())
    print(undirected_graph.edges())

    nx.draw(undirected_graph)
    plot_graph(undirected_graph, os.path.join(args.figures_path, add_name + 'est_graph_rename.pdf'))

def recipr_t4_graph():
    """
    Rename the nodes of the T4 reciprocity graph.
    """
    # Create new folders
    args = TrainOptions().parse()
    args.exp_name = 'ebay_pc_recipr_t4'
    args.flow_name = 'cop_flow'
    args.alpha_indep = 0.05
    add_name = 'flow'

    # Create Folders
    args = create_folders(args)

    # Read pickled graph
    undirected_graph = nx.read_gpickle(os.path.join(os.path.join(args.figures_path, add_name + 'est_graph.pickle')))
    print(nx.info(undirected_graph))

    # Show the nodes
    print(undirected_graph.nodes())
    print(undirected_graph.edges())
    
    # Rename nodes
    mapping = {'log_concessions': 'C1', 
               'log_opp_concessions': 'C2', 
               'log_offr_price': 'P', 
               'log_time_since_offer': 'T', 
               'log_hist': 'H1', 
               'log_opp_hist': 'H2'}
    undirected_graph = nx.relabel_nodes(undirected_graph, mapping)

    # Show the nodes
    print(undirected_graph.nodes())
    print(undirected_graph.edges())

    nx.draw(undirected_graph)
    plot_graph(undirected_graph, os.path.join(args.figures_path, add_name + 'est_graph_rename.pdf'))


def timing_graph():
    """
    Rename the nodes of the timing graph.
    """

    # Create new folders
    args = TrainOptions().parse()
    args.exp_name = 'ebay_pc_timing'
    args.flow_name = 'cop_flow'
    args.alpha_indep = 0.05
    add_name = 'flow'

    # Create Folders
    args = create_folders(args)

    # Read pickled graph
    undirected_graph = nx.read_gpickle(os.path.join(os.path.join(args.figures_path, add_name + 'est_graph.pickle')))
    print(nx.info(undirected_graph))

    # Show the nodes
    print(undirected_graph.nodes())
    print(undirected_graph.edges())
    
    # Rename nodes
    mapping = {'log_concessions': 'C1', 
               'log_opp_concessions': 'C2', 
               'log_offr_price': 'P', 
               'log_time_since_offer': 'T', 
               'log_hist': 'H1', 
               'log_opp_hist': 'H2',
               'log_response_time': 'R1',
               'log_opp_response_time': 'R2'}
    undirected_graph = nx.relabel_nodes(undirected_graph, mapping)

    # Show the nodes
    print(undirected_graph.nodes())
    print(undirected_graph.edges())

    nx.draw(undirected_graph)
    plot_graph(undirected_graph, os.path.join(args.figures_path, add_name + 'est_graph_rename.pdf'))

if __name__ == "__main__":
    recipr_graph()
    recipr_t4_graph()
    timing_graph()