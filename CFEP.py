# -*- coding: utf-8 -*-
"""
This module implements CFEP detection.
"""
from __future__ import print_function

import array
from decimal import Decimal

import numbers
import warnings

import networkx as nx
import numpy as np
import random
import time
from tqdm import tqdm

from Flow_group_status import Status


__PASS_MAX = -1
__MIN = 0.0000001


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def CFEP_detection(graph,after_graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_state=None,
                        pattern=None,
                        show_process=False):
    
    """Find flow group from two network snapshots


    Parameters
    ----------
    graph : networkx.Graph
        the networkx graph which will be decomposed
    part_init : dict, optional
        the algorithm will start using this partition of the nodes. It's a
        dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    resolution :  double, optional
        Will scale the expectation of flow change rate, default to 1.
    randomize :  see the definition of check_random_state(seed)
    pattern : str
        'Co-increasing', 'Positive Co-stable', 'Co-decreasing', 'Negative Co-stable'
    show_process: Bool
        Show the status of CFEP detection
        
    

    Returns
    -------
    Flow group partition : list of dictionaries
        a list of partitions, ie dictionnaries where keys of the i+1 are the
        values of the i. and where keys of the first are the nodes of graph

    Raises
    ------
    TypeError
        If the graph is not a networkx.Graph

    """
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    # Properly handle random state, eventually remove old `randomize` parameter
    # NOTE: when `randomize` is removed, delete code up to random_state = ...
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If shouldn't randomize, we set a fixed seed to get determinisitc results
        if randomize is False:
            random_state = 0

    # We don't know what to do if both `randomize` and `random_state` are defined
    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_state = check_random_state(random_state)

    # special case, when there is no link
    # the best partition is everyone in its flow group
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]
    
    obj_func=None
    obj_dir=None
    if pattern=='Co-increasing':
        obj_func='fraction_max'
        obj_dir='max'
    elif pattern=='Positive Co-stable':
        obj_func='fraction_min'
        obj_dir='min'
    elif pattern=='Co-decreasing':
        obj_func='fraction_min'
        obj_dir='min'
    elif pattern=='Negative Co-stable':
        obj_func='fraction_max'
        obj_dir='max'
    else:
        return 'Please use a correct CFEP type'
    
    print(pattern)
    current_graph = graph.copy()
    after_graph_copy = after_graph.copy()
    
    status = Status()
    status_after = Status()
    
    status.init(current_graph, weight, part_init)
    status_after.init(after_graph_copy, weight, part_init)
    
    __one_level(current_graph, after_graph_copy, status, status_after, weight, resolution, random_state, obj_func, obj_dir,show_process)


    return status.node2com




def stopping_criteria(stable_Q_list, check_continue_epoch=100):
    if len(stable_Q_list)<check_continue_epoch:
        return True
    else:
        last_10=stable_Q_list[check_continue_epoch*-1:]
        last_10=np.array(last_10)
        last_10_TF=last_10==last_10[-1]
        if False not in last_10_TF and last_10[-1]==1:
            return False
        else:
            return True
        
def __one_level(graph, graph_after, status, status_after, weight_key, resolution, random_state, obj_func, obj_dir,show_process):
    
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status, status_after, resolution, obj_func,obj_dir, graph, graph_after,weight_key)
    new_mod = cur_mod
    stable_Q_list=[]
    all_edge_weight_graph_after=[]
    for edge in graph_after.edges():
        graph_after_edge_weight=graph_after[edge[0]][edge[1]][weight_key]
        if graph_after_edge_weight>0:
            all_edge_weight_graph_after.append(graph_after_edge_weight)
    
    while stopping_criteria(stable_Q_list, check_continue_epoch=2):
        t = time.time()
        cur_mod = new_mod
        nb_pass_done += 1
            
        for node in __randomize(graph.nodes(), random_state):
            com_node = status.node2com[node]
            neigh_communities = __neighcom(node, graph, status, weight_key)
            neigh_communities_after = __neighcom(node, graph_after, status_after, weight_key)

            best_increase=0
            best_com = com_node
            
            for com, dnc in __randomize(neigh_communities.items(), random_state):
                before_modified_mod = __modularity_in_neighcom(status,status_after, graph, graph_after, 
                                                                   weight_key,[com,com_node],obj_dir,resolution,obj_func,all_edge_weight_graph_after)
                __remove(node, com_node,
                         neigh_communities.get(com_node, 0.), 
                         neigh_communities_after.get(com_node, 0.), 
                         status, status_after)
                
                __insert(node, com,
                         neigh_communities.get(com, 0.), 
                         neigh_communities_after.get(com, 0.), status, status_after)

                after_modified_mod = __modularity_in_neighcom(status,status_after, graph, graph_after, 
                                                                  weight_key,[com,com_node],obj_dir,resolution,obj_func,all_edge_weight_graph_after)
                modified_increase=after_modified_mod-before_modified_mod

                if  modified_increase-best_increase>__MIN: 
                    best_increase = modified_increase
                    best_com = com

                __remove(node, com,
                     neigh_communities.get(com, 0.), 
                     neigh_communities_after.get(com, 0.), 
                     status, status_after)
                __insert(node, com_node,
                         neigh_communities.get(com_node, 0.), 
                         neigh_communities_after.get(com_node, 0.), 
                         status, status_after)
                    
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), 
                     neigh_communities_after.get(com_node, 0.), 
                     status, status_after)
            __insert(node, best_com,
                     neigh_communities.get(best_com, 0.), neigh_communities_after.get(best_com, 0.), status, status_after)
            

        
        new_mod = __modularity(status,status_after, resolution, obj_func,obj_dir, graph, graph_after, weight_key)
        if show_process:
            print(str(nb_pass_done)+' iteration  '+'Q:'+str(new_mod)+'  '+f'coast:{time.time() - t:.4f}s')
        if new_mod - cur_mod < __MIN:
            stable_Q_list.append(1)
        else:
            stable_Q_list.append(0)
            
        
            
def __modularity_in_neighcom(status,status_after, graph, graph_after, weight_key,com_before_after,obj_dir,resolution,obj_func,all_edge_weight_graph_after):
    modified_mod = 0. 
    mean_edge_weight_graph_after=sum(all_edge_weight_graph_after)/len(all_edge_weight_graph_after)
    std_edge_weight_graph_after=np.std(all_edge_weight_graph_after)
    max_edge_weight_graph_after=max(all_edge_weight_graph_after)
    max_edge_weight_graph_after_zscore=(max_edge_weight_graph_after-mean_edge_weight_graph_after)/std_edge_weight_graph_after
    min_edge_weight_graph_after=min(all_edge_weight_graph_after)
    min_edge_weight_graph_after_zscore=(min_edge_weight_graph_after-mean_edge_weight_graph_after)/std_edge_weight_graph_after
    
    for community in com_before_after:
        # find nodes in current community
        node_list_in_com=[]
        for node,corr_com in status.node2com.items():
            if corr_com==community:
                node_list_in_com.append(node)

        if len(node_list_in_com)>1:
            for link in graph.subgraph(node_list_in_com).edges(data=True):

                graph_degree=graph.degree(weight=weight_key)[link[0]]*graph.degree(weight=weight_key)[link[1]]
                graph_after_degree=graph_after.degree(weight=weight_key)[link[0]]*graph_after.degree(weight=weight_key)[link[1]]
                
                graph_after_edge_weight=graph_after[link[0]][link[1]][weight_key]
                graph_edge_weight=graph[link[0]][link[1]][weight_key]

                if graph_after_edge_weight>__MIN and graph_after_degree>__MIN:
                    
                    edge_weight_adjust=1
                    current_edge_zscore=((graph_after_edge_weight-mean_edge_weight_graph_after)/std_edge_weight_graph_after)
                    
                    if graph_after_edge_weight>mean_edge_weight_graph_after:
                        edge_weight_adjust=1+current_edge_zscore/max_edge_weight_graph_after_zscore
                    else:
                        edge_weight_adjust=(current_edge_zscore-min_edge_weight_graph_after_zscore)/(-min_edge_weight_graph_after_zscore)
                    if obj_dir=='max':
                        modified_mod+=edge_weight_adjust*(graph_edge_weight/graph_after_edge_weight-resolution*(graph_degree/graph_after_degree)/(status.total_weight/status_after.total_weight))
                        
                    if obj_dir=='min':
                        modified_mod+=edge_weight_adjust*(resolution*(graph_degree/graph_after_degree)/(status.total_weight/status_after.total_weight)-graph_edge_weight/graph_after_edge_weight)
                        
                else:
                    if obj_dir=='min' and obj_func=='fraction_max':
                        modified_mod+=-1
                    else:
                        modified_mod+=0

            
    return modified_mod
    
    
def __neighcom(node, graph, status, weight_key):
    """
    Compute the flow group in the neighborhood of node in the graph given
    with the decomposition node2com
    """
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights


def __remove(node, com, weight, weight_after, status,status_after):
    """ Remove node from flow group com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1
    
    status_after.degrees[com] = (status_after.degrees.get(com, 0.)
                           - status_after.gdegrees.get(node, 0.))
    status_after.internals[com] = float(status_after.internals.get(com, 0.) -
                                  weight_after - status_after.loops.get(node, 0.))
    status_after.node2com[node] = -1


def __insert(node, com, weight, weight_after, status, status_after):
    """ Insert node into flow group and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))
    
    status_after.node2com[node] = com
    status_after.degrees[com] = (status_after.degrees.get(com, 0.) +
                           status_after.gdegrees.get(node, 0.))
    status_after.internals[com] = float(status_after.internals.get(com, 0.) +
                                  weight_after + status_after.loops.get(node, 0.))


def __modularity(status,status_after, resolution ,obj_func,obj_dir, graph, graph_after, weight_key):
    all_edge_weight_graph_after=[]
    for edge in graph_after.edges():
        graph_after_edge_weight=graph_after[edge[0]][edge[1]][weight_key]
        if graph_after_edge_weight>0:
            all_edge_weight_graph_after.append(graph_after_edge_weight)
    if obj_func=='fraction_max':
        return __modularity_fraction(status,status_after, resolution, graph, graph_after, weight_key,obj_dir,all_edge_weight_graph_after)
    elif obj_func=='fraction_min':
        return __modularity_fraction_min(status,status_after, resolution, graph, graph_after, weight_key,obj_dir,all_edge_weight_graph_after)
    else:
        print('The objective function is not in the list.')

def __modularity_fraction(status,status_after, resolution,graph, graph_after, weight_key,obj_dir,all_edge_weight_graph_after):

    result = 0.
    mean_edge_weight_graph_after=sum(all_edge_weight_graph_after)/len(all_edge_weight_graph_after)
    std_edge_weight_graph_after=np.std(all_edge_weight_graph_after)
    max_edge_weight_graph_after=max(all_edge_weight_graph_after)
    max_edge_weight_graph_after_zscore=(max_edge_weight_graph_after-mean_edge_weight_graph_after)/std_edge_weight_graph_after
    min_edge_weight_graph_after=min(all_edge_weight_graph_after)
    min_edge_weight_graph_after_zscore=(min_edge_weight_graph_after-mean_edge_weight_graph_after)/std_edge_weight_graph_after
    for community in set(status.node2com.values()):
        # find nodes in current community
        node_list_in_com=[]
        for node,corr_com in status.node2com.items():
            if corr_com==community:
                node_list_in_com.append(node)
                
        internal_fraction=0
        if len(node_list_in_com)>1:
            for link in graph.subgraph(node_list_in_com).edges(data=True):
                if (link[1]!=link[0]):

                    graph_degree=graph.degree(weight=weight_key)[link[0]]*graph.degree(weight=weight_key)[link[1]]
                    graph_after_degree=graph_after.degree(weight=weight_key)[link[0]]*graph_after.degree(weight=weight_key)[link[1]]
                    graph_after_edge_weight=graph_after[link[0]][link[1]][weight_key]
                    graph_edge_weight=graph[link[0]][link[1]][weight_key]
                    if graph_after[link[0]][link[1]][weight_key]>__MIN and graph_after_degree>__MIN:
                        edge_weight_adjust=1
                        current_edge_zscore=((graph_after_edge_weight-mean_edge_weight_graph_after)/std_edge_weight_graph_after)
                        if graph_after_edge_weight>mean_edge_weight_graph_after:
                            edge_weight_adjust=1+current_edge_zscore/max_edge_weight_graph_after_zscore
                        else:
                            edge_weight_adjust=(current_edge_zscore-min_edge_weight_graph_after_zscore)/(-min_edge_weight_graph_after_zscore)
                        result+=edge_weight_adjust*(graph_edge_weight/graph_after_edge_weight-resolution*(graph_degree/graph_after_degree)/(status.total_weight/status_after.total_weight))

                    else:
                        if obj_dir=="min":
                            result+=-1
                        else:
                            result+=0

    return result

def __modularity_fraction_min(status,status_after, resolution,graph, graph_after, weight_key,obj_dir,all_edge_weight_graph_after):

    result = 0.
    mean_edge_weight_graph_after=sum(all_edge_weight_graph_after)/len(all_edge_weight_graph_after)
    std_edge_weight_graph_after=np.std(all_edge_weight_graph_after)
    max_edge_weight_graph_after=max(all_edge_weight_graph_after)
    max_edge_weight_graph_after_zscore=(max_edge_weight_graph_after-mean_edge_weight_graph_after)/std_edge_weight_graph_after
    min_edge_weight_graph_after=min(all_edge_weight_graph_after)
    min_edge_weight_graph_after_zscore=(min_edge_weight_graph_after-mean_edge_weight_graph_after)/std_edge_weight_graph_after
    for community,community_after in zip(set(status.node2com.values()),set(status_after.node2com.values())):
        # find nodes in current community
        node_list_in_com=[]
        for node,corr_com in status.node2com.items():
            if corr_com==community:
                node_list_in_com.append(node)
                
        internal_fraction=0
        if len(node_list_in_com)>1:
            for link in graph.edges(data=True):
                if (link[0] in node_list_in_com) & (link[1] in node_list_in_com) & (link[1]!=link[0]):

                    graph_degree=graph.degree(weight=weight_key)[link[0]]*graph.degree(weight=weight_key)[link[1]]
                    graph_after_degree=graph_after.degree(weight=weight_key)[link[0]]*graph_after.degree(weight=weight_key)[link[1]]
                    
                    graph_after_edge_weight=graph_after[link[0]][link[1]][weight_key]
                    graph_edge_weight=graph[link[0]][link[1]][weight_key]
                    
                    if graph_after[link[0]][link[1]][weight_key]>__MIN and graph_after_degree>__MIN:
                        edge_weight_adjust=1
                        current_edge_zscore=((graph_after_edge_weight-mean_edge_weight_graph_after)/std_edge_weight_graph_after)
                        if graph_after_edge_weight>mean_edge_weight_graph_after:
                            edge_weight_adjust=1+current_edge_zscore/max_edge_weight_graph_after_zscore
                        else:
                            edge_weight_adjust=(current_edge_zscore-min_edge_weight_graph_after_zscore)/(-min_edge_weight_graph_after_zscore)
                        result+=edge_weight_adjust*(resolution*(graph_degree/graph_after_degree)/(status.total_weight/status_after.total_weight)-graph_edge_weight/graph_after_edge_weight)
#                         print(status.total_weight,status_after.total_weight)

                    else:
                        result+=0

    return result



def __randomize(items, random_state):
    """Returns a List containing a random permutation of items"""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items

def generate_test_graph(sizes, probs, com_m, seed):

    random.seed(seed)

    def _random_subset(seq, m, rng):
        """Return m unique elements from seq.

        This differs from random.sample which can return repeated
        elements if seq holds repeated elements.

        Note: rng is a random.Random or numpy.random.RandomState instance.
        """
        targets = set()
        while len(targets) < m:
            x = random.choice(seq)
            targets.add(x)
        return targets

    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    # edit each partition to be scale-free by BA model
    for partition,m in zip(G.graph['partition'],com_m):
        partition=list(partition)
        n=len(partition)
        initial_graph=None

        if m < 1 or m >= n:
            raise nx.NetworkXError(
                f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
            )
        
        star_graph_nodes=random.sample(partition,m+1) #select m+1 nodes to build star graph
        
        # Select the remaining nodes after building the star network for later addition to the star network
        rest_node=partition.copy()
        for node in star_graph_nodes:
            rest_node.remove(node)

        hub=random.choice(star_graph_nodes)
        star_graph_nodes.remove(hub)
#         print(hub)

        # build star graph
        G.add_edges_from([(hub,connected_node) for connected_node in star_graph_nodes])
        G_star=nx.Graph()
        G_star.add_edges_from([(hub,connected_node) for connected_node in star_graph_nodes])

        # List of existing nodes, with nodes repeated once for each adjacent edge
        repeated_nodes = [n for n, d in G_star.degree(partition) for _ in range(d)]

        # Start adding the other n - m nodes.
        for source in rest_node:
            # Now choose m unique nodes from the existing nodes
            # Pick uniformly from repeated_nodes (preferential attachment)
            targets = _random_subset(repeated_nodes, m, seed)
            # Add edges to m nodes from the source.
            G.add_edges_from(zip([source] * m, targets))
            # Add one node to the list for each new edge just created.
            repeated_nodes.extend(targets)
            # And the new node "source" has m edges to add to the list.
            repeated_nodes.extend([source] * m)
    return G

def sum_weight(g_list):
    weight_list=[]
    for g in g_list:
        total_weight=0
        for edge in g.edges(data=True):
            total_weight+=edge[2]['weight']
        weight_list.append(total_weight)

    return weight_list
def get_negative_edge(G_t1,G_t2):
    # general weight change rate is 1
    # in positive pattern, select edges by determining if F_g2/F_g1 - 1
    negative_edge_G_t1=[]
    negative_edge_G_t2=[]
    G_list=sum_weight([G_t1,G_t2])
    for edge1 in tqdm(G_t1.edges()):
        if G_t2[edge1[0]][edge1[1]]['weight']==0:
            continue
        edge_change_rate=G_t1[edge1[0]][edge1[1]]['weight']/G_t2[edge1[0]][edge1[1]]['weight']-G_list[0]/G_list[1]
        if edge_change_rate<0:
            negative_edge_G_t1.append([edge1[0],edge1[1],G_t1[edge1[0]][edge1[1]]['weight']])
            negative_edge_G_t2.append([edge1[0],edge1[1],G_t2[edge1[0]][edge1[1]]['weight']])

    G_n1=nx.Graph()
    G_n2=nx.Graph()

    G_n1.add_weighted_edges_from([(edge[0],edge[1],edge[2]) for edge in negative_edge_G_t1])
    G_n2.add_weighted_edges_from([(edge[0],edge[1],edge[2]) for edge in negative_edge_G_t2])
    return G_n1,G_n2

def std_topology(weekday_G,weekend_G):
    G_d1_copy=weekday_G.copy()
    G_d2_copy=weekend_G.copy()
    G_d2_edge_list=list(G_d2_copy.edges())
    G_d1_edge_list=list(G_d1_copy.edges())
    # print(G_d2_edge_list)
    added_edge_d1=[]
    added_edge_d2=[]
    for edge in tqdm(G_d2_edge_list):
        if edge not in G_d1_edge_list and (edge[1],edge[0]) not in G_d1_edge_list:
            added_edge_d1.append((edge[1],edge[0],0))

    G_d1_copy.add_weighted_edges_from(added_edge_d1)

    for edge in tqdm(G_d1_edge_list):
        if edge not in G_d2_edge_list and (edge[1],edge[0]) not in G_d2_edge_list:
            added_edge_d2.append((edge[1],edge[0],0))

    G_d2_copy.add_weighted_edges_from(added_edge_d2)
    
    return G_d1_copy,G_d2_copy

def get_positive_graph(G_t1,G_t2):
    # general weight change rate is 1
    # in positive pattern, select edges by determining if F_g2/F_g1 - 1
    positive_edge_G_t1=[]
    positive_edge_G_t2=[]
    rate=[]
    G_list=sum_weight([G_t1,G_t2])
#     print(G_list[0]/G_list[1])
    for edge1 in tqdm(G_t1.edges()):
        if G_t2[edge1[0]][edge1[1]]['weight']==0:
            positive_edge_G_t1.append([edge1[0],edge1[1],G_t1[edge1[0]][edge1[1]]['weight']])
            positive_edge_G_t2.append([edge1[0],edge1[1],G_t2[edge1[0]][edge1[1]]['weight']])
        else:
            edge_change_rate=G_t1[edge1[0]][edge1[1]]['weight']/G_t2[edge1[0]][edge1[1]]['weight']-G_list[0]/G_list[1]
            if edge_change_rate>0:
                rate.append(edge_change_rate+1)
                positive_edge_G_t1.append([edge1[0],edge1[1],G_t1[edge1[0]][edge1[1]]['weight']])
                positive_edge_G_t2.append([edge1[0],edge1[1],G_t2[edge1[0]][edge1[1]]['weight']])
    
    G_p1=nx.Graph()
    G_p2=nx.Graph()

    G_p1.add_weighted_edges_from([(edge[0],edge[1],edge[2]) for edge in positive_edge_G_t1])
    G_p2.add_weighted_edges_from([(edge[0],edge[1],edge[2]) for edge in positive_edge_G_t2])
    return G_p1,G_p2

def get_negative_graph(G_t1,G_t2):
    # general weight change rate is 1
    # in positive pattern, select edges by determining if F_g2/F_g1 - 1
    negative_edge_G_t1=[]
    negative_edge_G_t2=[]
    G_list=sum_weight([G_t1,G_t2])
#     print(G_list[0]/G_list[1])
    for edge1 in tqdm(G_t1.edges()):
        if G_t2[edge1[0]][edge1[1]]['weight']==0:
            continue
        edge_change_rate=G_t1[edge1[0]][edge1[1]]['weight']/G_t2[edge1[0]][edge1[1]]['weight']-G_list[0]/G_list[1]
        if edge_change_rate<0:
            negative_edge_G_t1.append([edge1[0],edge1[1],G_t1[edge1[0]][edge1[1]]['weight']])
            negative_edge_G_t2.append([edge1[0],edge1[1],G_t2[edge1[0]][edge1[1]]['weight']])

    G_n1=nx.Graph()
    G_n2=nx.Graph()

    G_n1.add_weighted_edges_from([(edge[0],edge[1],edge[2]) for edge in negative_edge_G_t1])
    G_n2.add_weighted_edges_from([(edge[0],edge[1],edge[2]) for edge in negative_edge_G_t2])
    return G_n1,G_n2