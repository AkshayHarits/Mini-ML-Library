import graphviz
from my_ml_lib.nn.autograd import Value # Assuming Value is here
import numpy as np

from my_ml_lib.nn import Sequential, Linear, ReLU, Module

# --- Implement Graph Traversal ---
def get_all_nodes_and_edges(root_node: Value):
    """
    Performs a backward traversal from the root_node
    to find all unique Value nodes and the directed edges connecting them
    in the computation graph.
    ...
    """
    # --- Step 1: Initialize Sets ---
    nodes = set() # Store unique Value objects
    edges = set() # Store unique edge tuples (parent, child)
    visited = set() # Keep track of nodes already processed

    # --- Step 2: Implement DFS Traversal Function ---
    def build_graph(v):
        # a) Check if 'v' has already been visited
        if v in visited:
            return
        
        # b) Add v to the visited set
        visited.add(v)
        
        # c) Add v to the nodes set
        nodes.add(v)
        
        # d) Iterate through the parents of v
        for parent in v._prev:
            # i) Add the edge (parent, v) to the edges set
            edges.add((parent, v))
            # ii) Recursively call the helper function on the parent
            build_graph(parent)

    # --- Step 3: Start Traversal ---
    build_graph(root_node)

    # --- Step 4: Return Results ---
    return nodes, edges
# --- End Implementation ---


# --- [Boilerplate Graph Drawing Function: 2276-2321] ---
def draw_dot(root_node: Value, format='svg', rankdir='LR'):
    """Generates a visualization of the computation graph using graphviz."""
    assert rankdir in ['LR', 'TB']
    nodes, edges = get_all_nodes_and_edges(root_node)
    dot = graphviz.Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        uid = str(id(n))
        data_str = f"shape={n.data.shape}" if hasattr(n, 'data') and isinstance(n.data, np.ndarray) and n.data.ndim > 0 else f"{getattr(n, 'data', '?'):.4f}"
        grad_str = f"shape={n.grad.shape}" if hasattr(n, 'grad') and isinstance(n.grad, np.ndarray) and n.grad.ndim > 0 else f"{getattr(n, 'grad', '?'):.4f}"
        label_str = f" | {getattr(n, 'label', '')}" if getattr(n, 'label', '') else ""
        
        node_label = f"{{ {label_str} | data {data_str} | grad {grad_str} }}"
        dot.node(name=uid, label=node_label, shape='record')

        op = getattr(n, '_op', '')
        if op:
            op_uid = uid + op
            dot.node(name=op_uid, label=op)
            dot.edge(op_uid, uid)

    for n1, n2 in edges: # Edge: parent (n1) -> child (n2)
        parent_uid = str(id(n1))
        child_op = getattr(n2, '_op', '')
        if child_op:
            child_op_uid = str(id(n2)) + child_op
            dot.edge(parent_uid, child_op_uid)
            
    return dot

if __name__ == "__main__":
    print("\n--- Visualization Example (Tiny MLP) ---")
    
    # For reproducible random weights in the Linear layer
    np.random.seed(42)

    # 1. Define the tiny MLP: Sequential(Linear(2, 3), ReLU())
    model = Sequential(
        Linear(in_features=2, out_features=3, bias=True),
        ReLU()
    )
    
    # 2. Create a sample input (batch size = 1, features = 2)
    x_input = Value(np.array([[1.0, -2.0]]), label='x_input')

    # 3. Perform one forward pass
    y_output = model(x_input)
    y_output.label = 'y_output'

    # 4. Create a dummy loss (e.g., sum of outputs) to backpropagate from
    # This is just to populate all the gradients for visualization
    loss = y_output.sum()
    loss.label = 'dummy_loss'

    print("Running backward pass on example...")
    try:
        # 5. Run backward pass
        loss.backward()
    except Exception as e:
        print(f"Note: Backward pass failed in example: {e}. Visualization might still work.")

    print("Generating example computation graph...")
    # 6. Draw the graph starting from the final loss node
    dot_graph = draw_dot(loss)

    if dot_graph:
        try:
            output_filename = 'computation_graph'
            # We updated this to 'png' in a previous step
            dot_graph.render(output_filename, view=False, format='png')
            print(f"Example graph saved as {output_filename}.png")
            print("Please rename this file to 'computational_graph.png' for your submission.")
        except graphviz.backend.execute.ExecutableNotFound:
            print("\n--- Graphviz Error ---")
            print("Graphviz executable not found. Visualization not saved.")
            print("Please install Graphviz (from www.graphviz.org)")
            print("and ensure the 'dot' command is available in your system's PATH.")
            print("----------------------\n")
        except Exception as e:
            print(f"An error occurred during graph rendering: {e}")
    else:
        print("Graph generation failed (likely due to traversal error).")