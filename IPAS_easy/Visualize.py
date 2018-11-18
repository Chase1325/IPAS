"""Visualize ThreatFields, Paths, Sensor Locations, GridWorlds"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def draw_threat_field(env, threat_field, x_res=100, y_res=100, colorbar=True):
    """Draw a 3D threat field and return an axis object which can be used to update plot

    ax = draw_threat_field(env=env, threat_field=threat_field, x_res = 200, y_res = 200)

    x_res and y_res are optional arguments to change the resolution of the plot, default = 100"""
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data
    X = np.linspace(0, env.x_size, x_res)
    Y = np.linspace(0, env.y_size, y_res)
    X, Y = np.meshgrid(X, Y)

    Z = threat_field.threat_value(X, Y)

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)

    # Customize the z axis
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    if colorbar:
        fig.colorbar(surf, shrink=0.5, aspect=5)

    #plt.show(block=False)
    return ax


def draw_threat_field_2D(env, threat_field, ax=None, x_res=100, y_res=100, colorbar=True):
    """Draw a 2D threat field and return an axis object to further modify plot

    ax_2d = draw_threat_field_2D(env=env, threat_field=threat_field)

    This function is useful for later plotting the path on top"""
    if not ax:
        plot, ax = plt.subplots(1, 1)

    # Make data
    X = np.linspace(0, env.x_size, x_res)
    Y = np.linspace(0, env.y_size, y_res)
    X, Y = np.meshgrid(X, Y)

    Z = threat_field.threat_value(X, Y)

    pcol = ax.pcolor(X, Y, Z, cmap=cm.jet)
    if colorbar:
        plt.colorbar(pcol)
    #plt.show(block=False)
    return ax


def draw_path(ax, path, color='white'):
    """Add a path to an existing plot. Used in combination with draw_threat_field_2D.
    Typical usage:

    ax_2d = draw_threat_field_2D(env=env, threat_field=threat_field)
    draw_path(ax=ax_2d, path=path)

    path is a list of Nodes obtained after:

    goal_vertex_found = Astar(graph=graph, start_vertex=start_vertex, goal_vertex=goal_vertex)
    path = [goal_vertex_found.node]
    reconstruct_path(goal_vertex_found, path)"""
    x = [n.pos_x for n in path]
    y = [n.pos_y for n in path]

    ax.plot(x, y, markersize=8, color=color,
            marker='o', markeredgewidth=2.0, markeredgecolor='black')

    #plt.show(block=False)
