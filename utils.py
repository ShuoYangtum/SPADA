import re
import pandas as pd
from collections import defaultdict

def reverse_dependency_dict(dep_dict):
    reversed_dict = defaultdict(list)

    for key, values in dep_dict.items():
        for value in values:
            reversed_dict[value].append(key)

    for key in dep_dict.keys():
        reversed_dict.setdefault(key, [])
    reversed_dict=dict(reversed_dict)
    if "" in reversed_dict:
        reversed_dict.pop("")
        
    return reversed_dict


def extract_keys(s):
    match = re.search(r'\[(.*?)\]', s)
    if match:
        content = match.group(1)
        return [pair.split("->")[0].strip() for pair in content.split(", ")]
    return []
    
def check_keys(nodes_and_edges, keys):
    for key in keys:
        if key not in nodes_and_edges:
            print(f"The dependency of feature {key} is not found in response.")
            return False
    for node in nodes_and_edges:
        if node not in keys:
            print(f"The feature {node} in response is not found in the table.")
            return False
        for edge in nodes_and_edges[node]:
            if edge not in keys and edge:
                print(f"The dependency {edge} of feature {node} in response is not found in the table.")
                return False
    return True


def save_to_csv(data, filename="output.csv"):
    df = pd.DataFrame(data)  
    df.to_csv(filename, index=False)  
    print(f"Data save to {filename}.")