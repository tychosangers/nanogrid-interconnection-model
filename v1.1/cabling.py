# ============================================================================
#  cabling.py
#  Interconnection cable length: a minimum spanning tree (MST) over the
#  nanogrid coordinates, multiplied by a routing factor.
#  Run this file to print the lengths and plot the cable network.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from configuration import NANOGRID_COORDINATES, CABLE_ROUTING_FACTOR, NANOGRID_NAMES, COST_COLOURS

METERS_PER_DEG = 111320      # metres per degree of latitude


def distance_m(lat1, lon1, lat2, lon2):
    """Straight-line distance between two coordinates [m] (flat-earth approximation)."""
    mean_lat = np.radians((lat1 + lat2) / 2)
    dy = (lat2 - lat1) * METERS_PER_DEG
    dx = (lon2 - lon1) * METERS_PER_DEG * np.cos(mean_lat)
    return np.sqrt(dx**2 + dy**2)


def cable_mst_edges(coords):
    """Minimum spanning tree over the coordinates (Prim's algorithm).

    Returns one (i, j, length_m) tuple per cable link.
    """
    n = len(coords)
    in_tree = [0]
    edges = []
    while len(in_tree) < n:
        best_d = None
        best_i = None
        best_j = None
        for i in in_tree:
            for j in range(n):
                if j not in in_tree:
                    d = distance_m(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                    if best_d is None or d < best_d:
                        best_d = d
                        best_i = i
                        best_j = j
        edges.append((best_i, best_j, best_d))
        in_tree.append(best_j)
    return edges


def cable_length_mst(coords):
    """Total straight-line length of the minimum spanning tree [m]."""
    total_length = 0.0
    for _, _, d in cable_mst_edges(coords):
        total_length += d
    return total_length


def cable_length_routed(coords):
    """Cable length used for costing: MST length times the routing factor [m]."""
    return cable_length_mst(coords) * CABLE_ROUTING_FACTOR


def coords_to_local_m(coords):
    """Convert coordinates to east and north offsets from the first nanogrid [m]."""
    lat0, lon0 = coords[0]
    cos_lat = np.cos(np.radians(lat0))
    xs = [(lon - lon0) * METERS_PER_DEG * cos_lat for lat, lon in coords]
    ys = [(lat - lat0) * METERS_PER_DEG for lat, lon in coords]
    return xs, ys


def plot_cable_network(coords):
    """Map of the nanogrids and the cable links with their straight-line lengths."""
    xs, ys = coords_to_local_m(coords)
    edges = cable_mst_edges(coords)
    cable_colour = COST_COLOURS["Cable capex"]

    fig, ax = plt.subplots(figsize=(6, 6))

    for i, j, d in edges:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=cable_colour, linewidth=2, zorder=1)
        mx = (xs[i] + xs[j]) / 2
        my = (ys[i] + ys[j]) / 2
        ax.annotate(f"{d:.1f} m", (mx, my), textcoords="offset points",
                    xytext=(0, 6), ha="center", color=cable_colour)

    ax.scatter(xs, ys, color="dimgrey", s=80, zorder=2)
    for k in range(len(coords)):
        ax.annotate(NANOGRID_NAMES[k], (xs[k], ys[k]), textcoords="offset points", xytext=(8, 8))

    total = cable_length_mst(coords)
    routed = cable_length_routed(coords)
    ax.set_title(f"Interconnection cable network\n"
                 f"straight-line MST {total:.1f} m, routed {routed:.1f} m (factor {CABLE_ROUTING_FACTOR:.2f})")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    total = cable_length_mst(NANOGRID_COORDINATES)
    routed = cable_length_routed(NANOGRID_COORDINATES)
    print(f"Straight-line MST length: {total:.1f} m")
    print(f"Routed length (factor {CABLE_ROUTING_FACTOR:.2f}): {routed:.1f} m")
    plot_cable_network(NANOGRID_COORDINATES)
