import re
from collections import defaultdict

def build_adjacency_matrix(graph_content):
   

    # Split input into individual sections
    sections = graph_content.strip().split('\n\n')
    
    # Initialize adjacency matrix
    size = 10  # Including 'Q' (0) and sub-questions 1-9
    adj_matrix = [[0] * size for _ in range(size)]
    
    # Track edges to determine in-degree
    edges = defaultdict(list)
    nodes_present = set()

    # Parse the graph content from each section
    for section in sections:
        lines = section.strip().split('\n')
        for line in lines:
            if '->' in line:
                # Handle multiple steps in one line
                parts = line.split('->')
                for i in range(len(parts) - 1):
                    src_part = parts[i].strip()
                    dest_part = parts[i + 1].strip()

                    # Extract numbers from strings
                    src_match = re.search(r'\d+', src_part)
                    src = int(src_match.group()[0]) if src_match and 'Q' not in src_part else 0

                    dest_nodes = []
                    for x in re.split(r',\s*', dest_part):
                        dest_match = re.search(r'\d+', x)
                        if dest_match:
                            dest_nodes.append(int(dest_match.group()[0]))
                        elif 'Q' in x:
                            dest_nodes.append(0)

                    for dest_node in dest_nodes:
                        edges[src].append(dest_node)
                        nodes_present.add(src)
                        nodes_present.add(dest_node)

    # Fill the adjacency matrix based on edges
    for src in edges:
        for dest in edges[src]:
            adj_matrix[src][dest] = 1

    # Ensure 'Q' does not point to sub-questions
    for dest in range(1, size):
        adj_matrix[0][dest] = 0

    # Nodes with no outgoing edges that have appeared in the graph point to 'Q'
    for node in nodes_present:
        if node != 0 and (node not in edges or all(dest == 0 for dest in edges[node])):
            adj_matrix[node][0] = 1

    return adj_matrix

