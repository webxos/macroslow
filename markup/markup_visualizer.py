import plotly.graph_objects as go

class MarkupVisualizer:
    def render_3d_graph(self, graph_data: Dict):
        """Render a 3D graph of the transformation process using Plotly."""
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        # Node positions (simple layout for demo)
        node_x, node_y, node_z = [], [], []
        for i, node in enumerate(nodes):
            node_x.append(i)
            node_y.append(i % 2)
            node_z.append(i // 2)

        # Edge lines
        edge_x, edge_y, edge_z = [], [], []
        for edge in edges:
            from_idx = next(i for i, n in enumerate(nodes) if n["id"] == edge["from"])
            to_idx = next(i for i, n in enumerate(nodes) if n["id"] == edge["to"])
            edge_x.extend([node_x[from_idx], node_x[to_idx], None])
            edge_y.extend([node_y[from_idx], node_y[to_idx], None])
            edge_z.extend([node_z[from_idx], node_z[to_idx], None])

        # Create Plotly traces
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=2, color="#888"),
            hoverinfo="none",
            mode="lines"
        )

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode="markers+text",
            text=[node["label"] for node in nodes],
            marker=dict(size=10, color=[1 if node["group"] == "input" else 2 for node in nodes], colorscale="Viridis")
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="Markdown-to-Markup Transformation Graph",
                            showlegend=False,
                            scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z"))
                        ))

        fig.write_to_html("transformation_graph.html")
