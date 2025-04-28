import networkx as nx
from utils import extract_keys, check_keys
from scipy.sparse.linalg import eigs
import cvxpy as cp

def build_graph_from_LLM_response(response, keys):
    response=[res for res in response.split('\n') if res]
    nodes_and_edges={key.split(": ")[0]:extract_keys(key) for key in response}

    if check_keys(nodes_and_edges, keys):
        return nodes_and_edges


def build_graph(dependencies):
    G = nx.DiGraph()
    for node, parents in dependencies.items():
        for parent in parents:
            G.add_edge(parent, node)
    return G

def remove_cycles(G, method='pagerank'):
    if method=='pagerank':
        dag = G.copy()
        try:
            nx.find_cycle(dag, orientation='original') 
        except nx.NetworkXNoCycle:
            print("图已是 DAG，无需修改")
            return dag 
        
        pagerank = nx.pagerank(dag)

        cycles = list(nx.simple_cycles(dag))
    
        edges_to_remove = set()
        for cycle in cycles:
            min_edge = None
            min_importance = float('inf')
            
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                importance = pagerank.get(u, 0) + pagerank.get(v, 0)
                if importance < min_importance:
                    min_importance = importance
                    min_edge = (u, v)
            
            if min_edge:
                edges_to_remove.add(min_edge)
    
        dag.remove_edges_from(edges_to_remove)
    elif method=='ilp':
        edges = list(G.edges)
        n = len(edges)
        
        x = cp.Variable(n, boolean=True)
        
        obj = cp.Minimize(cp.sum(x))
        
        constraints = []
        for cycle in nx.simple_cycles(G):
            cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
            indices = [edges.index(e) for e in cycle_edges if e in edges]
            constraints.append(cp.sum(x[indices]) >= 1)  
        
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.GUROBI)  
    
        edges_to_remove = [edges[i] for i in range(n) if x.value[i] > 0.5]
        G.remove_edges_from(edges_to_remove)
        dag=G
    elif method=='dfs':
        dag = G.copy()
        
        try:
            cycle = nx.find_cycle(dag, orientation='original')
        except nx.NetworkXNoCycle:
            return dag  
    
        while True:
            try:
                cycle = nx.find_cycle(dag, orientation='original')
              
                u, v, _ = min(cycle, key=lambda edge: G.degree(edge[1]))
                dag.remove_edge(u, v)
            except nx.NetworkXNoCycle:
                break
    return dag


def generate_dependencies(DAG):
    """从 DAG 生成新的 dependencies 结构"""
    dependencies = {}
    
    for node in DAG.nodes():
        predecessors = list(DAG.predecessors(node))
        dependencies[node] = predecessors if predecessors else ['']
    
    return dependencies