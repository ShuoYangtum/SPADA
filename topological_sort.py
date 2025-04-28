import networkx as nx

def topological_layers_from_dag(DAG, keys=None):
    reverse_dependencies = {node: set(DAG.predecessors(node)) for node in DAG.nodes}

    first_layer = [node for node in DAG.nodes if len(reverse_dependencies[node]) == 0 or reverse_dependencies[node]=={''}]

    layers = [first_layer]  
    visited = set(first_layer)

    while len(visited) < len(DAG.nodes):
        next_layer = []
        for node in DAG.nodes:
            if node not in visited and reverse_dependencies[node].issubset(visited):
                next_layer.append(node)

        if not next_layer:
            raise ValueError("Graph contains unreachable nodes or cycles!")

        layers.append(next_layer)
        visited.update(next_layer)
    for i in range(len(layers)):
        if "" in layers[i]:
            layers[i].remove("")
    if keys:
        check_if_all_features_is_in_layers(layers, keys)
    return layers

def check_if_all_features_is_in_layers(layers, keys):
    nodes=[]
    for layer in layers:
        for node in layer:
            nodes.append(node)
    nodes=set(nodes)
    assert len(nodes)==len(keys)
    for key in keys:
        if key not in nodes:
            print(f"The key {key} is not in your layers.")
            return False
    return True