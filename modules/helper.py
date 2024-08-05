import numpy as np
import pydot
from IPython.display import Image, display

def SSE(samples: np.ndarray, targets: np.ndarray|None=None, validate: bool=False):
    assert len(samples.shape) == 1

    if validate: 
        assert len(samples) == len(targets), f"Mismatch shape for inputs and labels, inputs shape {samples.shape} with labels shape {targets.shape}"
        mean = targets
    else:
        mean = np.mean(samples)

    return np.sum(np.square(samples - mean))

def MAE(samples: np.ndarray, targets: np.ndarray|None=None, validate: bool=False):
    assert len(samples.shape) == 1

    if validate: 
        assert len(samples) == len(targets), f"Mismatch shape for inputs and labels, inputs shape {samples.shape} with labels shape {targets.shape}"
        mean = targets
    else:
        mean = np.mean(samples)

    return np.mean(np.absolute(samples - mean))

def MSE(samples: np.ndarray, targets: np.ndarray|None=None, validate: bool=False):
    assert len(samples.shape) == 1

    if validate: 
        assert len(samples) == len(targets), f"Mismatch shape for inputs and labels, inputs shape {samples.shape} with labels shape {targets.shape}"
        mean = targets
    else:
        mean = np.mean(samples)

    return np.mean(np.square(samples - mean))

def visualize_tree(root_node):
    graph = pydot.Dot("test", graph_type="digraph")

    node_counter = 0
    graph.add_node(pydot.Node(node_counter, label=root_node.label))
    node_counter += 1

    queue = [{'parent_index': 0, 'branch':item[0], 'curr_node':item[1]} for item in root_node.branches.items()]

    while len(queue) > 0:
        curr_item = queue.pop(0)
        if curr_item['curr_node'].metric is None:
            graph.add_node(pydot.Node(node_counter, label=curr_item['curr_node'].label))
        else:
            graph.add_node(pydot.Node(node_counter, label=f"{curr_item['curr_node'].label}: {curr_item['curr_node'].split_val}"))

            for branch, value in curr_item['curr_node'].branches.items():    
                queue.append({'parent_index': node_counter, 'branch': branch, 'curr_node': value})
            
        graph.add_edge(pydot.Edge(curr_item['parent_index'], node_counter, label=curr_item['branch']))
            
        node_counter += 1

    plt = Image(graph.create_png())
    display(plt)