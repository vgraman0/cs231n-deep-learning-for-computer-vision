
import numpy as np
import matplotlib.pyplot as plt


def plot_contours(ax, f, x_range=(-3.0, 3.0), y_range=(-3.0, 3.0), levels=50, grid_size=200):
    """
    Plot filled contours of a 2D scalar function f(theta).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    f : callable
        Function that takes an array of shape (..., 2) and returns scalar values.
    x_range, y_range : tuple
        (min, max) for the x and y axes.
    levels : int
        Number of contour levels.
    grid_size : int
        Resolution of the grid used to evaluate f.

    Returns
    -------
    cs : QuadContourSet
        The contourf handle.
    """
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(xs, ys)
    points = np.stack([X, Y], axis=-1)
    Z = f(points)

    cs = ax.contourf(X, Y, Z, levels=levels, alpha=0.9)
    ax.contour(X, Y, Z, levels=levels, linewidths=0.5)
    ax.set_xlabel("theta_0")
    ax.set_ylabel("theta_1")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal", adjustable="box")
    return cs


def plot_trajectory(ax, trajectory, label=None, linewidth=2):
    """
    Plot an optimization trajectory on top of an existing contour plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    trajectory : array-like, shape (T, 2)
        Sequence of parameter vectors visited by the optimizer.
    label : str, optional
        Label for the legend.
    linewidth : float
        Width of the trajectory line.
    """
    traj = np.asarray(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], marker="o", label=label, linewidth=linewidth)
    # highlight start and end
    ax.scatter(traj[0, 0], traj[0, 1], marker="x", s=60)
    ax.scatter(traj[-1, 0], traj[-1, 1], marker="o", s=60)


def visualize_optimization(f, trajectories, x_range=(-3.0, 3.0), y_range=(-3.0, 3.0),
                           levels=50, grid_size=200, figsize=(6, 6)):
    """
    Convenience helper to plot a loss surface and several trajectories.

    Parameters
    ----------
    f : callable
        Scalar loss function f(theta).
    trajectories : dict[str, array-like]
        Mapping from name to trajectory array of shape (T, 2).
    x_range, y_range : tuple
        Ranges for axes.
    levels : int
        Number of contour levels.
    grid_size : int
        Resolution of the grid.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_contours(ax, f, x_range=x_range, y_range=y_range, levels=levels, grid_size=grid_size)
    for name, traj in trajectories.items():
        plot_trajectory(ax, traj, label=name)
    ax.legend()
    ax.set_title("Optimization trajectories")
    return fig, ax
