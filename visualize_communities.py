import sys
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons

def load_graphs(path, max_nodes_limit=50000):
    print(f"Loading graphs from {path}...")
    # Load on CPU to avoid errors if local machine has no GPU
    graphs = torch.load(path, map_location='cpu', weights_only=False)
    
    # Filter out massive communities that would freeze the NetworkX layout
    filtered_graphs = []
    for g in graphs:
        n = g.num_nodes if hasattr(g, 'num_nodes') else (g.x.size(0) if g.x is not None else 0)
        if n > 0 and n <= max_nodes_limit:
            filtered_graphs.append(g)
            
    # Sort them by size (ascending) so the smallest ones are shown first
    filtered_graphs.sort(key=lambda g: getattr(g, 'num_nodes', g.x.size(0) if g.x is not None else 0))
    
    print(f"{len(graphs)} total communities found. Kept {len(filtered_graphs)} communities with <= {max_nodes_limit} nodes.")
    return filtered_graphs

class GraphVisualizer:
    def __init__(self, graphs):
        self.graphs = graphs
        self.current_idx = 0
        
        # Checkbox state
        self.show_edges = {
            'reply': False,
            'retweet': False,
            'mention': False
        }
        
        # Colors associated with edge types
        self.colors = {
            'reply': 'blue',
            'retweet': 'green',
            'mention': 'red'
        }
        
        # Create matplotlib interface
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2, left=0.25)
        
        # --- Next / Previous buttons ---
        axprev = plt.axes([0.35, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.55, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_graph)
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_graph)
        
        # --- Checkboxes ---
        axcheck = plt.axes([0.02, 0.4, 0.18, 0.15])
        axcheck.set_title("Edge Types")
        self.check = CheckButtons(
            axcheck, 
            ('reply (blue)', 'retweet (green)', 'mention (red)'),
            (True, True, True)
        )
        self.check.on_clicked(self.toggle_edges)
        
        # Draw the first graph
        self.draw_graph()

    def get_edge_list(self, data, attr_name):
        """Extract edge list from a PyTorch edge_index"""
        edge_index = getattr(data, attr_name, None)
        if edge_index is not None and edge_index.size(0) == 2 and edge_index.size(1) > 0:
            return list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        return []

    def draw_graph(self):
        self.ax.clear()
        
        if not self.graphs:
            self.ax.set_title("No graphs found.")
            plt.draw()
            return
            
        data = self.graphs[self.current_idx]
        
        if hasattr(data, 'num_nodes'):
            num_nodes = data.num_nodes
        elif data.x is not None:
            num_nodes = data.x.size(0)
        else:
            num_nodes = 0
            
        edges_reply = self.get_edge_list(data, 'edge_index_reply')
        edges_retweet = self.get_edge_list(data, 'edge_index_retweet')
        edges_mention = self.get_edge_list(data, 'edge_index_mention')
        
        # We use NetworkX to compute the layout
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        
        # Add ALL edges to the NetworkX graph so the layout considers the full structure
        G.add_edges_from(edges_reply)
        G.add_edges_from(edges_retweet)
        G.add_edges_from(edges_mention)
        
        # Also include the 'general' edge_index if it exists (to stabilize the layout)
        general_edges = self.get_edge_list(data, 'edge_index')
        G.add_edges_from(general_edges)

        # Compute positions
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=self.ax, node_color='lightgray', node_size=150, edgecolors='gray')
        
        # Draw edges based on checkboxes
        if self.show_edges['reply'] and edges_reply:
            nx.draw_networkx_edges(G, pos, ax=self.ax, edgelist=edges_reply, edge_color=self.colors['reply'], 
                                   arrows=True, width=1.2, arrowsize=10, connectionstyle="arc3,rad=0.1")
        if self.show_edges['retweet'] and edges_retweet:
            nx.draw_networkx_edges(G, pos, ax=self.ax, edgelist=edges_retweet, edge_color=self.colors['retweet'], 
                                   arrows=True, width=1.2, arrowsize=10, connectionstyle="arc3,rad=0.2")
        if self.show_edges['mention'] and edges_mention:
            nx.draw_networkx_edges(G, pos, ax=self.ax, edgelist=edges_mention, edge_color=self.colors['mention'], 
                                   arrows=True, width=1.2, arrowsize=10, connectionstyle="arc3,rad=0.3")

        # Display counters
        title = (f"Community {self.current_idx + 1} / {len(self.graphs)}\n"
                 f"Nodes: {num_nodes} | Reply: {len(edges_reply)} | Retweet: {len(edges_retweet)} | Mention: {len(edges_mention)}")
        self.ax.set_title(title, pad=10)
        self.ax.axis('off')
        
        plt.draw()

    def next_graph(self, event):
        self.current_idx = (self.current_idx + 1) % len(self.graphs)
        self.draw_graph()

    def prev_graph(self, event):
        self.current_idx = (self.current_idx - 1) % len(self.graphs)
        self.draw_graph()

    def toggle_edges(self, label):
        if 'reply' in label:
            self.show_edges['reply'] = not self.show_edges['reply']
        elif 'retweet' in label:
            self.show_edges['retweet'] = not self.show_edges['retweet']
        elif 'mention' in label:
            self.show_edges['mention'] = not self.show_edges['mention']
        self.draw_graph()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: You must specify the path to the .pt file")
        print("Usage: python visualize_communities.py <path_to_communities.pt>")
        sys.exit(1)
        
    path = sys.argv[1]
    graphs = load_graphs(path)
    
    vis = GraphVisualizer(graphs)
    plt.show()
