#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:00:54 2019

@author: pade
This is a toolbox of methods for Machine Learning on graphs. Some methods are
implemented from
[1] Liang - A nonparametric classification method based on k-associated graphs - 2011
"""

import numpy as np
import data2graph as dg


# -----------------------------------------------------------------------------
# Unsupervised methods
# -----------------------------------------------------------------------------


def partition(laplacian):
    """
    Spectral clustering of the graph with Laplacian matrix @laplacian.
    """
    [E, V] = np.linalg.eig(laplacian)
    # Delete zero eigenvalue and eigenvector
    zero_ind = np.argmin(E)
    E_new = np.delete(E, zero_ind)
    V_new = np.delete(V, zero_ind, 1)
    # Find Fiedler vector
    fiedler_ind = np.argmin(E_new)
    fiedler_vec = V_new[:, fiedler_ind]           
    return fiedler_vec


# -----------------------------------------------------------------------------
# Supervised methods
# -----------------------------------------------------------------------------


def purity(adjacency, k):
    """
    Compute the purity of a digraph.
    @adjacency is the adjacency matrix.
    @k is an integer. It comes from the graph construction through the
      k-associated method.
    """
    degs = np.array([sum(row) for row in adjacency + adjacency.transpose()])
    deg_avg = sum(degs)/len(adjacency)
    return deg_avg/2/k
    

def kaog_classifier(data, labels, point, distance=np.linalg.norm, weight=False):
    """
    A Bayesian classifier based on the k-associated optimal graph (kaog)
    produced by the training data @data.
    """
    # This is the adjacency matrix of the graph from kaog and the corresponding
    # connected components.
    adjacency, k_values = dg.k_assoc_opt(data, labels, distance=np.linalg.norm, weight=weight)
    graph_comps = dg.find_components(adjacency)
    
    # -------------------------------------------------------------------------
    # Remove isolated vertices from the k-associated optimal graph and adjust
    # adjacency, data, labels, k-values and labeling of the nodes accordingly 
    
    # Identify isolated vertices...
    del_nodes = [comp[0] for comp in graph_comps if len(comp)<2]
    # ... and determine transformation of the node labeling.
    ind_trafo = lst_update(list(range(len(adjacency))), del_nodes)

    # Remove nodes in adjacency matrix
    adjacency = np.delete(adjacency, del_nodes, 0)
    adjacency = np.delete(adjacency, del_nodes, 1)

    # Remove data points
    data = np.delete(data, del_nodes, 0)
    for i in sorted(del_nodes, reverse=True):
        del labels[i]    

    # Remove components of size one and corresponding k-values
    graph_comps_new = []
    k_values_new = []
    for i in range(len(graph_comps)):
        if len(graph_comps[i])>1:
            graph_comps_new.append(graph_comps[i])
            k_values_new.append(k_values[i])
    graph_comps = graph_comps_new
    k_values = k_values_new

    # Relabel nodes after the deletion process
    for i in range(len(graph_comps)):
        for l in range(len(graph_comps[i])):
            graph_comps[i][l] = ind_trafo[graph_comps[i][l]]
    # -------------------------------------------------------------------------
    
    
    distances = np.argsort([distance(point - other) for other in data])
    pCond = []
    pComp = []
    pLambda = []
    # Compute maximal k-value and the denominator of Eq. (9) in ref. [1]
    k_max = max(k_values)
    p_normalization = pComp_denom(data, labels, k_max, point)
    # Determine the components that @point connects to along with their k-value
    connected_components = [comp for comp in zip(graph_comps, k_values)\
                            if any([index in comp[0] for index in distances[:k_max]])]
   
    
    # Iterate through components
    for component, k_value in connected_components:
        
        # Eq. 7 in ref [1] ----------------------------------------------------
        pCond.append(pCond_func(data, labels, component, k_value, point, distance=distance, weight=weight))
        #----------------------------------------------------------------------
        
        # Eq. 9 in ref [1] ----------------------------------------------------
        pComp.append(purity(adjacency[np.ix_(component, component)], k_value)/p_normalization)
        # ---------------------------------------------------------------------

     
    # Eq. 8 in ref[1] 
    pLambda = sum([pcon*pcom for pcon, pcom in zip(pCond, pComp)])

    # Finally, sum up the probabilities for each label (Eq.(10) in ref [1])----
    probabilities = {label: 0 for label in labels}
    connected_components = [comp[0] for comp in connected_components]
    for i, component in enumerate(connected_components):
        # Identify the label of the component
        try:
            label = labels[component[0]]
        except TypeError:
            label = labels[component]
        probabilities.setdefault(label, 0)
        if pLambda != 0:
            probabilities[label] += pCond[i]*pComp[i]/pLambda
        elif pLambda == 0 and (pCond[i]*pComp[i] == 0):
            probabilities[label] += 0
        elif pLambda == 0 and (pCond[i]*pComp[i] != 0):
            print('Zero Division Error!')
    
    return probabilities, adjacency, data, labels
    
    
def pCond_func(data, labels, component, k_value, point, distance=np.linalg.norm, weight=False):
    """
    Compute the conditioned probability from Eq.(7) in ref.[1].
    """
    order = np.argsort([distance(point - other) for other in data])
    neighbours = order[:k_value]
    return len(set(neighbours).intersection(set(component)))/k_value
    

def pComp_denom(data, labels, k_value, point, distance=np.linalg.norm, weight=False):
    """
    Compute the probability that @point is in the component @component
    according to Eq.(9) in ref.[1].
    """
    adjacency, k_values = dg.k_assoc_opt(data, labels, distance=np.linalg.norm, weight=False)    
    order = np.argsort([distance(point - other) for other in data])
    
    # Compute the denominator -----
    denominator = 0
    # Determine the components that @point connects to along with their k-value
    connected_components = [(comp, value) for comp, value in zip(dg.find_components(adjacency), k_values)\
                            if any([index in comp for index in order[:k_value]])]
    # Make list unique...
    connected_components_unique = []
    for value in connected_components:
        if value not in connected_components_unique:
            connected_components_unique.append(value)
    # ...and add the purities up.
    for component in connected_components_unique:
        denominator += purity(adjacency[np.ix_(component[0], component[0])], component[1]) 
    return denominator


# -----------------------------------------------------------------------------
# Auxilliary (matrix-)tools
# -----------------------------------------------------------------------------


def lst_update(lst, indices):
    """
    Compute index transformation when elements with indices @indices are
    removed from the list @lst."""
    trafo = {k:k for k in range(len(lst))}
    for index in sorted(indices, reverse=True):
        for i in range(index+1,len(lst)):
            trafo[i] -= 1
    for index in sorted(indices, reverse=True):
        del trafo[index]
    return trafo
        

def rem_isol(data, labels,  distance=np.linalg.norm, weight=False):
    """
    Removes isolated nodes from the graph obtained by the k-associated 
    optimal method."""
    adjacency, k_values = dg.k_assoc_opt(data, labels, distance=distance, weight=weight)
    conn_comps = dg.find_components(adjacency)
     # Remove isolated components
    ind_delete = [component[0] for component in conn_comps if len(component) < 2]
    data = np.delete(data, ind_delete, 0)
    for index in sorted(ind_delete, reverse=True):
        del labels[index]
    return data, labels
    

def Laplacian(adjacency):
    """
    Compute the Laplacian matrix of an adjacency matrix. Both given as
    square numpy arrays.
    """
    L = -adjacency
    k = 0
    for row in adjacency:
        L[k,k] = sum(row)
        k += 1
    return L


def is_weakly(adjacency, tol=10e-14):
    """
    Returns True if the corresponding digraph is weakly connected and False
    otherwise.
    @tol accounts for computational inaccuracy. Values below @tol are
     classified as zero.
    """
    # Symmetrize adjacency matrix
    adjacency = np.maximum( adjacency, adjacency.transpose())
    # Check, whether the Laplacian has a double zero eigenvalue
    laplacian = Laplacian(adjacency)
    [E, V] = np.linalg.eig(laplacian)
    E = np.delete(E, np.argmin(E))
    V = np.delete(V, np.argmin(E), 1)
    if min(E) > tol:
        return True
    return False


def nodeswitch(adjacency, i, j):
    """
    Given an adjacency matrix of a (di-)graph and two nodes i and j, the 
    adjacency matrix of the same graph is returned, except that nodes i and j 
    are interchanged.
    """
    adjacency[[i,j], :] = adjacency[[j,i], :]
    adjacency[:, [i,j]] = adjacency[:, [j,i]]
    return adjacency


def graph_components(adjacency):
    """
    Determines (weakly) connected components of the (di-)graph given by the
    adjacency matrix @adjacency. Return value is a list of lists, where each 
    sublist lists the nodes belonging to one connected component."""
    # First check, whether the graph is disconnected.
    if is_weakly(adjacency):
        print("Error: The graph is not disconnected.")
        return
    n = len(adjacency)
    adj_symm = np.maximum( adjacency, adjacency.transpose())
    # Entry i,j of @paths is nonzero iff there is a path from node j to node i.
    paths = sum([np.linalg.matrix_power(adj_symm,k) for k in range(n)])
    # Each list in @connections corresponds to nodes in one component.
    components = []
    nodes_sum, k = 0, 0
    # We can stop the iteration as soon as all graph components are found.
    while nodes_sum < sum(range(n)) and k < n:
        if list(np.nonzero(paths[k, :])[0]) in components:
            k += 1
        else:
            components.append(list(np.nonzero(paths[k, :])[0]))
            nodes_sum = sum([item for lst in components for item in lst])
            k += 1
    return components
  
    
def listoflistorder(lst):
    """
    Checks whether a list of lists is ordered in the following sense: No 
    entry of sublist lst[k] is smaller than all entries of sublist lst[k-1] for
    all 0 < k < len(lst).
    """
    for k in range(1,len(lst)):
        if any( [min(lst[k]) < value for value in lst[k-1]] ):
            return False
        else:
            k += 1
    return True
      

def disconn_diag(adjacency):
    """
    @disconn_diag(adjacency): brings the adjacency matrix of a disconnected 
    (di-)graph into block diagonal form by relabeling the nodes.
    """
    # First check, whether the graph is disconnected.
    if is_weakly(adjacency):
        print("Error: The graph is not disconnected.")
        return
    n = len(adjacency)
    # Step 1: Determine the graph's components --------------------------------
    # As the graph's components do not have to be strongly connected, use the
    # symmetrized adjacency matrix in order to find the (weak) components.
    adj_symm = np.maximum( adjacency, adjacency.transpose())
    # Entry i,j of @paths is nonzero iff there is a path from node j to node i.
    paths = sum([np.linalg.matrix_power(adj_symm,k) for k in range(n)])
    # Each list in @connections corresponds to nodes in one component.
    connections = []
    nodes_sum, k = 0, 0
    # We can stop the iteration as soon as all graph components are found.
    while nodes_sum < sum(range(n)) and k < n:
        if list(np.nonzero(paths[k, :])[0]) in connections:
            k += 1
        else:
            connections.append(list(np.nonzero(paths[k, :])[0]))
            nodes_sum = sum([item for lst in connections for item in lst])
            k += 1
    # Step 2: Relabel the nodes -----------------------------------------------
    # Now we can relabel the nodes according to the graph components.
    # First presort the node numbers list.
    connections = sorted(connections, key = lambda x: sum(x)/len(x) )
    # The permutations of the nodes are collected in this dictionary.
    index_change = {x:x for x in range(n)}
    k = 0
    while not listoflistorder(connections) and k+1 < len(connections):
        while max(connections[k]) > min([min(lst) for lst in connections[k+1:]]):
            # Switch maximal node number with global minimum,...
            adjacency = nodeswitch(adjacency, max(connections[k]), \
                                   min([min(lst) for lst in connections[k+1:]]))
            # ...store the swap in the permutation dictionary...
            index_change[max(connections[k])] = min([min(lst) for lst in connections[k+1:]])
            index_change[min([min(lst) for lst in connections[k+1:]])] = max(connections[k])
            # ...and make the same switch in the connections list.
            M = max(connections[k])
            connections[k].remove(M)
            connections[k].append(min([min(lst) for lst in connections[k+1:]]))
            aux_list = [lst.index(min(lst)) for lst in connections[k+1:]]
            list_number = np.argmin(aux_list) + k+1
            connections[list_number].remove(min(connections[list_number]))
            connections[list_number].append(M)
        k += 1      
    return adjacency, connections, index_change
