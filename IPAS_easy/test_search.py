"""Test out Threat and threat field classes"""

from Threat import GaussThreat, GaussThreatField
from Environment import XYEnvironment
import matplotlib.pyplot as plt
from Visualize import draw_threat_field, draw_threat_field_2D, draw_path
from Graph import XYNode, Vertex, Graph
from Search import Astar, reconstruct_path


def main():
    env = XYEnvironment(x_size=10, y_size=10, x_pts=30, y_pts=30)

    threat1 = GaussThreat(location=(2, 2), shape=(0.5, 0.5), intensity=5)
    threat2 = GaussThreat(location=(8, 8), shape=(1.0, 1.0), intensity=5)
    threat3 = GaussThreat(location=(8, 2), shape=(1.5, 1.5), intensity=5)
    threats = [threat1, threat2, threat3]

    threat_field = GaussThreatField(threats=threats, offset=2)

    threat4 = GaussThreat(location=(2, 8), shape=(0.5, 0.5), intensity=5)
    threat_field.add_threat(threat4)
    env.add_threat_field(threat_field)

    print(env)

    node_list = {}
    for gridpt in range(env.n_grid):
        mx = gridpt % env.n_grid_x
        my = int(gridpt / env.n_grid_x)
        new_node = XYNode(node_id=gridpt)
        new_node.pos_x = mx * env.grid_sep_x
        new_node.pos_y = my * env.grid_sep_y
        node_list[gridpt] = new_node
        print(new_node)

    graph = Graph(env=env)
    graph.add_vertex(node_list[0])
    start_vertex = graph.get_vertex(node_list[0])
    print("Start vertex: ", start_vertex)

    neighbor_list = graph.env.get_neighbors(start_vertex.node)
    print("neighbor_list")
    print(neighbor_list)
    for nbr in neighbor_list:
        print(nbr)
        graph.add_edge(start_vertex.node, nbr, 1.0)

    for neighbor in start_vertex.neighbors:
        print(neighbor)

    print("Run A* search")
    goal_vertex = Vertex(node=node_list[env.n_grid - 1])
    print(goal_vertex.node)

    goal_vertex_found = Astar(graph=graph, start_vertex=start_vertex, goal_vertex=goal_vertex)
    print(goal_vertex_found)

    path = [goal_vertex_found.node]
    reconstruct_path(goal_vertex_found, path)
    print("Path found :")
    for waypoint in path:
        print(waypoint)

    draw_threat_field(env=env, threat_field=threat_field)

    ax2 = draw_threat_field_2D(env=env, threat_field=threat_field)

    draw_path(ax=ax2, path=path)
    plt.show()


if __name__ == "__main__":
    main()