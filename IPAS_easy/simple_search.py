"""Test out Threat and threat field classes"""

from Threat import GaussThreat, GaussThreatField
from Environment import XYEnvironment
import matplotlib.pyplot as plt
from Visualize import draw_threat_field, draw_threat_field_2D, draw_path
from Graph import XYNode, Vertex, Graph
from Search import Astar, reconstruct_path


def main():
    # Create a 2D environment
    env = XYEnvironment(x_size=10, y_size=10, x_pts=30, y_pts=30)

    # Create and add some threats to a field
    threat1 = GaussThreat(location=(2, 2), shape=(0.5, 0.5), intensity=5)
    threat2 = GaussThreat(location=(8, 8), shape=(1.0, 1.0), intensity=5)
    threat3 = GaussThreat(location=(8, 2), shape=(1.5, 1.5), intensity=5)
    threat4 = GaussThreat(location=(2, 8), shape=(0.5, 0.5), intensity=5)
    threats = [threat1, threat2, threat3, threat4]
    threat_field = GaussThreatField(threats=threats, offset=2)

    # Add the threat_field to the Environment
    env.add_threat_field(threat_field)
    print(env)

    # Create the Graph based on our environment
    graph = Graph(env=env)

    # Add a starting node and vertex to the graph to begin the search at.
    # Use first node in the environment as start node
    start_node = XYNode(node_id=0, pos_x=0, pos_y=0)
    graph.add_vertex(start_node)
    start_vertex = graph.get_vertex(start_node)
    print("Start vertex: ", start_vertex)

    # Use last node in environment as goal node
    goal_x, goal_y = env.get_location_from_gridpt(env.n_grid - 1)
    goal_node = XYNode(env.n_grid - 1, goal_x, goal_y)
    goal_vertex = Vertex(node=goal_node)
    print("Goal Node: ", goal_vertex.node)

    print("Run A* search")
    goal_vertex_found = Astar(graph=graph, start_vertex=start_vertex, goal_vertex=goal_vertex)
    print(goal_vertex_found)

    # Generate the path from the discovered goal vertex
    path = [goal_vertex_found.node]
    reconstruct_path(goal_vertex_found, path)
    print("Path found :")
    for waypoint in path:
        print(waypoint)

    # Plot the threat field in 3D
    draw_threat_field(env=env, threat_field=threat_field)

    # Plot the threat field as 2D heat map and add path on top
    ax2 = draw_threat_field_2D(env=env, threat_field=threat_field)
    draw_path(ax=ax2, path=path)

    plt.show()


if __name__ == "__main__":
    main()