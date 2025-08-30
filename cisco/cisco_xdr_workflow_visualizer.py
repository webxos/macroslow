# xdr_workflow_visualizer.py: Visualizes Cisco XDR workflows with DUNES CORE SDK
# CUSTOMIZATION POINT: Update graph_data for specific telemetry flows
from dunes_visualizer import DunesVisualizer

class XDRVisualizer:
    def __init__(self):
        self.visualizer = DunesVisualizer()

    def create_workflow_graph(self, telemetry_type: str):
        """Create a 3D graph for telemetry workflow."""
        graph_data = {
            "nodes": [
                {"id": "xdr_input", "label": f"Cisco XDR {telemetry_type}", "group": "input"},
                {"id": "maml", "label": "MAML Workflow", "group": "process"},
                {"id": "mu", "label": "MARKUP Receipt", "group": "output"}
            ],
            "edges": [
                {"from": "xdr_input", "to": "maml", "label": "Process"},
                {"from": "maml", "to": "mu", "label": "Generate Receipt"}
            ]
        }
        self.visualizer.render_3d_graph(graph_data)

# Example usage
if __name__ == "__main__":
    visualizer = XDRVisualizer()
    visualizer.create_workflow_graph("Endpoint")