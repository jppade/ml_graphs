#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:50:32 2019

@author: pade
This is a toolbox for transforming datasets into graphs. The datasets should be
given as numpy arrays. The resulting graphs can sub-sequently be used for
Machine Learning with the module ML_graphs.py.
"""

import numpy as np

# -----------------------------------------------------------------------------
# Unsupervised methods
# -----------------------------------------------------------------------------


def knnGraph(data, k, distance=np.linalg.norm, weight=False):
    """
    @knnGraph(data, k) returns the adjacency matrix of the graph built by 
    the k-nearest neighbors method from @data.
    @data is a numpy (m,n)-array where each row is a data point from R^n.
    @k is an integer.
    """
    m, n = np.shape(data)
    Adjacency = np.zeros((m,m))
    for i in range(m):
        if weight:
            for l, other in enumerate(data):
                try:
                    Adjacency[l, i] = 1/distance(data[i,:] - other)
                except ZeroDivisionError:
                    Adjacency[l, i] = 0
        else:    
            distances_ordered = np.argsort([distance(data[i,:] - other) for other in data])
            # Discard the smallest entry as it corresponds to the point itself
            Adjacency[distances_ordered[1:k+1], i] = 1
        del distances_ordered            
    return Adjacency


def eradiusGraph(data, eps, distance=np.linalg.norm, weight=False):
    """
    @eradiusGraph(data, eps) returns the adjacency matrix of the graph built 
    by the epsilon-radius network method.
    @data is a numpy (m,n)-array where each row is a data point from R^n.
    @eps is a positive number.
    """
    m, n = np.shape(data)
    Adjacency = np.zeros((m,m))
    for i in range(m):
        if weight:
            # In this case, the resulting adjacency matrix is weighted according
            # to the distance between the data points
            for l, other in enumerate(data):
                if distance(data[i,:] - other) < eps and l != i:
                    try:
                        Adjacency[l, i] = 1/distance(data[i,:] - other)
                    except ZeroDivisionError:
                        Adjacency[l, i] = 0
        else:
            Adjacency[[l for l, other in enumerate(data) \
                   if distance(data[i,:] - other) < eps and l!=i], i] = 1
    return Adjacency
    
    
def knn_erCombined(data, k, eps, distance=np.linalg.norm, weight=False):
    """
    @knn_erCombined(data, k, eps) returns the adjacency matrix of the graph
    built by a combination of the knn- and the epsilon-radius network method.
    In this method, the maximal degree in the graph is k.
    @data is a numpy (m,n)-array where each row is a data point from R^n.
    @k is an integer.
    @eps is a positive number.
    """
    m, n = np.shape(data)
    Adjacency = np.zeros((m,m))
    for i in range(m):
        # All indices which are in an eps-ball around data i
        er_array = [l for l, other in enumerate(data) \
                   if distance(data[i,:] - other) < eps and l!=i]
        if weight:
            # In this case, the resulting adjacency matrix is weighted
            # according to the distance between the data points
            if len(er_array) > k:
                order = np.argsort([distance(data[i,:] - other) for other in data])
                for l in range(k):
                    try:
                        Adjacency[order[l+1], i] = 1/(distance(data[i,:] - data[er_array[l+1]]))
                    except ZeroDivisionError:
                        Adjacency[order[l+1], i] = 0
                del order
            else:
                for l, other in enumerate(data):
                    if distance(data[i,:] - other) < eps and l != i:
                        try:
                            Adjacency[l, i] = 1/distance(data[i,:] - other)
                        except ZeroDivisionError:
                            Adjacency[l, i] = 0                            
        else:
            if len(er_array) > k:
                order = np.argsort([distance(data[i,:] - other) for other in data])
                Adjacency[order[1:k+1], i] = 1
                del order
            else:
                 Adjacency[[l for l, other in enumerate(data) \
                       if distance(data[i,:] - other) < eps and l!=i], i] = 1
    return Adjacency
            

# -----------------------------------------------------------------------------
# Supervised methods
# -----------------------------------------------------------------------------


def k_assoc(data, labels, k, distance=np.linalg.norm, weight=False):
    """
    @k_assoc(data, labels, k) returns the adjacency matrix of the digraph 
    built  by a method similar to the k-nearest neighbors method. The 
    difference is that here, a connection between two data points through 
    k-nearest neighbours is only established if the two labels are identical. 
    Hence, the out-degree of a node is in general smaller or equal to k.
    @data is a numpy (m,n)-array where each row is a data point from R^n.
    @labels is a list of m labels associated to the data.
    @k is an integer defining the (maximal) number of neighbours of each node.
    """
    m, n = np.shape(data)
    Adjacency = np.zeros((m,m))
    # dictionary of k nearest neighbours (independent of the label)
    incidence_dict = adj2list(knnGraph(data, k, distance=distance))
    for key in incidence_dict.keys():
        # Only account for neighours with the same label
        incidence_dict[key] = [value for value in incidence_dict[key] if labels[key]==labels[value]]
        if not weight:
            Adjacency[key, incidence_dict[key]] = 1
        else:
            for neighbour in incidence_dict[key]:
                try:
                    Adjacency[key, neighbour] = distance(data[neighbour, :] - data[key, :])
                except ZeroDivisionError:
                    Adjacency[key, neighbour] =  0                    
    return Adjacency
    

def k_assoc_opt(data, labels, distance=np.linalg.norm, weight=False):
    """
    @k_assoc_opt(data, labels) returns the adjacency matrix of the digraph 
    optimized by using the k-associated graph method. In contrast to the
    k-associated method itself, it is parameterless.
    @data is a numpy (m,n)-array where each row is a data point from R^n.
    @labels is a list of m labels associated to the data.
    """
    # Initiate the algorithm with the k-associated graph for k=1 --------------
    k = 1
    G_opt = k_assoc(data, labels, k, distance=distance, weight=weight)
    n = len(G_opt)
    # Determine connected components of G_opt.
    conn_comps_opt = find_components(G_opt)
    # At the first stage, all components have the same value k=1.
    k_values = [k for component in conn_comps_opt]
    # Compute the average degree of the network and an artifical previous 
    # degree in order to get the while loop running.
    deg_avg = 2*sum(sum(G_opt))/n
    last_avg = deg_avg - deg_avg/k -1

    # Iterate through the different values of parameter k ---------------------
    while deg_avg - last_avg > deg_avg/k and k < 2*n:
        k += 1
        # Compute the k-associated graph with incremented parameter k...
        G = k_assoc(data, labels, k, distance=distance, weight=weight)
        # ...and the corresponding connected components.
        conn_comps_G = find_components(G)
        
        # Update the optimal graph if the criterion @substitute is met --------
        for component in conn_comps_G:
            # Determine the components of @conn_comps_opt which are subcomponents
            # of @component in the new k-associated graph for k+1...
            subcomponents = [comp for comp in conn_comps_opt \
                             if set(comp).issubset((set(component)))]
            # ...and the associated k-values. 
            k_values_sub = [k_values[l] for l,comp in enumerate(conn_comps_opt) \
                             if set(comp).issubset((set(component)))]
            # This is the condition for substituting components.
            substitute = all([purity(G[np.ix_(component, component)],k)\
                              > purity(G_opt[np.ix_(component_old, component_old)], k_value)\
                              for component_old, k_value in zip(subcomponents, k_values_sub)])
            if substitute:
                # Remove connections from the graph which will be optimal...
                for component_old in subcomponents:
                    G_opt[component_old, :] = 0
                    G_opt[:, component_old] = 0
                # ... and substitute components with higher purity
                G_opt[component, :] = G[component, :]
                G_opt[:, component] = G[:, component]
                
        # Update the list of k-values -----------------------------------------
        aux = []
        for component in find_components(G_opt):
            # If the component was present at previous stage, don't change k
            if component in conn_comps_opt:
                aux.append(k_values[conn_comps_opt.index(component)])
            # Otherwise take the actual k
            else:
                aux.append(k)
                
        # Set values for the next iteration -----------------------------------
        # Compute new average degree and optimal graph's components for the
        # next iteration of the while-condition.
        last_avg, deg_avg = deg_avg, 2*sum(sum(G))/n
        conn_comps_opt = find_components(G_opt)
        k_values = aux
        
    return G_opt, k_values


# -----------------------------------------------------------------------------
# Auxiliary functions
# -----------------------------------------------------------------------------


def adj2list(adj):
    """
    @adj2list(adjacency) returns an incidence list for the graph given by
    the adjacency matrix @adj. Formally, the return value is a dictionary where
    each key stands for a node and the corresponding value is a list of
    (incoming) connections.
    @adj is a square numpy array.
    """
    adj_list = {}
    for i, row in enumerate(adj):
        adj_list[i] = [k for k, value in enumerate(row) if value != 0]
    return adj_list        
        

def denoise(data, labels, k, adjacency, distance=np.linalg.norm, weight=False):
    """
    Denoises the data set @data in the following sense: From the data build
    the k-associated graph and remove isolated nodes, as they most probably 
    correspond to noisy data.
    """
    if not adjacency:
        adjacency = k_assoc(data, labels, k, distance=distance, weight=weight)
    # Determine connected components of G_opt and store as list of lists
    conn_comps = find_components(adjacency)
    # Remove isolated components
    ind_delete = [component[0] for component in conn_comps if len(component) < 2]
    data = np.delete(data, ind_delete, 0)
    for index in sorted(ind_delete, reverse=True):
        del labels[index]
    return data, labels
    

def purity(adjacency, k):
    """
    Compute the purity of a digraph.
    @adjacency is the adjacency matrix of the (di-)graph given as a numpy array
    @k is an integer. It comes from the graph construction through the
      k-associated method.
    """
    degs = np.array([sum(row) for row in adjacency+adjacency.transpose()])
    deg_avg = sum(degs)/len(adjacency)
    return deg_avg/2/k


def Laplacian(adjacency):
    """
    Returns the Laplacian matrix of an adjacency matrix as a numpy array.
    @adjacency is the adjacency matrix of the (di-)graph given as a numpy array
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
    @adjacency is the adjacency matrix of the (di-)graph given as a numpy array
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


def find_components(adjacency):
    """
    Find connected components of a digraph given by the adjacency matrix
    @adjacency which is a square numpy array.
    """
    n = len(adjacency)
    # For connected components we can consider the symmetric adjacency matrix
    adjacency = adjacency + adjacency.transpose()
    # Begin with node 0.
    index = {0}
    # Collect all visited nodes in this set.
    nodes = index
    # Count components and collect them in a dictionary.
    num_component = 0
    components = {num_component:index}
    # Iterate through nodes until all are visited.
    while len(nodes) < n:
        index = get_predecessors(adjacency, index)
        # If all predecessors are already identified in the last step, pro-
        # ceed to the next component
        if all([(connections in components[num_component]) for connections in index]):
            remaining_indices = {ind for ind in range(n) if ind not in nodes}
            if remaining_indices:
                index = {list(remaining_indices)[0]}
                num_component += 1
                components[num_component] = index
                nodes = nodes.union(index)
        # Otherwise, add the nodes from @index to the component
        else:
            nodes = nodes.union(index)
            components[num_component] = set(components[num_component]).union(index)     
    components = [list(component) for component in list(components.values())]                         
    return components


def get_predecessors(adjacency, nodes):
    """
    Returns the predecessors of the nodes from the set @nodes in the graph
    given by the adjacency matrix @adjacency. Return value is a set.
    """
    connections = []
    for node in nodes:
        connections += [i for i,x in enumerate(adjacency[node, :]) if x != 0]
    return set(connections)
