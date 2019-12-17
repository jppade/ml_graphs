#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:53:43 2019

@author: pade
This module contains tools for visualization and testing of the data2graph and
the ML_graphs module. Run the main in order for testing and visualization.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ML_graphs as ml


def genNormal(mu, sigma, dim, num):
    """
    Generate @num numpy - arrays of length @dim, sampled from a normal distribution,
    with expectation @mu and variance @sigma.
    """
    data = np.zeros((num, dim))
    for k in range(num):
        data[k, :] = np.random.normal(mu, sigma, dim)
    return data


def plot_digraph(adjacency, data, labels, node_values):
    """
    Plots a digraph from an adjacency matrix using the networkx module. The
    nodes are organized in the plane according to the tuples given by @data. 
    The nodes are colored according to their label, given by @labels. The
    dictionary node_values with node names as keys is used for plotting.
    @adjacency is the adjacency matrix of the (di-)grgraphaph given as a numpy array
    @data is a numpy (m,2)-array where each row is a data point from R^2.
    @labels is a list of m labels associated to the data.
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        print("SizeError: The adjacency matrix must be square.")
        return
    elif data.shape[0] != adjacency.shape[0]:
        print("SizeError: The number of data points must coincide with the",\
              " number of graph's nodes.")
        return
    elif data.shape[0] != len(labels):
        print("SizeError: The number of data points must coincide with the",\
              " number of labels.")
        return
    plt.figure()
    
    # Set nodes and (directed) edges ------------------------------------------
    num_nodes = len(adjacency)
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i, pos=data[i])
        for k in range(num_nodes):
            if adjacency[i, k] > 0:
                G.add_edge(k, i)
    pos = nx.get_node_attributes(G, 'pos')
    labeldict = dict({(i, key) for i, key in enumerate(node_values.keys())})
    # -------------------------------------------------------------------------
    
    # Set node colors according to the labels ---------------------------------
    colors = {0:'blue', 1:'red', 2:'green', 3:'magenta', 4:'cyan', 5:'black',\
              6: 'yellow'}
    labellist = list(set(labels))
    node_colors  = [colors[labellist.index(label)] for label in labels]
    # -------------------------------------------------------------------------
    
    # Draw network ------------------------------------------------------------
    edges = G.edges()
    nx.draw_networkx_labels(G, pos, labels=labeldict, font_size=10, font_weight='normal')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edges=edges, alpha=0.4, arrowsize=20)
    plt.show()
    # -------------------------------------------------------------------------
    
    
def genTrainingData(dim_training_data, num_data, mean, variance):
    """
    Generates normally distributed training data.
    @dim_training_data is the dimension of the data points.
    @num_data is the number of data per label.
    @mean is a list of expectations. So the length corresponds to the number of
    datasets with different labels.
    @variance is the corresponding list of variances.
    """    
    training_data = genNormal(mean[0], variance[0], dim_training_data, num_data)
    labels = [0 for k in range(num_data)]
    for i in range(1, len(mean)):
        training_data = np.concatenate((training_data, genNormal(mean[i], variance[i], dim_training_data, num_data)))
        labels = labels + [i for k in range(num_data)]
    return training_data, labels
        

if __name__ == '__main__':
    
    
    training_data, labels = genTrainingData(2, 20, [0.1, 0.2, 0.3, 0.15], [0.1, 0.1, 0.05, 0.03])
    # Set a point which is to be classified.
    point = np.array([0.2, 0.24])
    
    a, adjacency, data, labels = ml.kaog_classifier(training_data, labels, point, distance=np.linalg.norm, weight=False)
    colors = {0:'blue', 1:'red', 2:'green', 3:'magenta', 4:'cyan', 5:'black',6: 'yellow'}
    for k in range(len(a)):
        a[colors[k]] = a.pop(k)
    print('The probabilities for the different classes are: ', a)
    plot_digraph(adjacency, data, labels, {})
    plt.plot(point[0], point[1], 'ko', markersize=14)
    
    
