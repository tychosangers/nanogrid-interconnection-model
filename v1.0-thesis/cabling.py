import numpy as np
import matplotlib.pyplot as plt
from configuration import Tinos_coordinates, Cable_routing_factor


def distance_m(lat1, lon1, lat2, lon2):        # straight-line distance between two coordinates [m]
    meters_per_deg = 111320                     # metres per degree of latitude (about constant)
    mean_lat = np.radians((lat1 + lat2) / 2)    # a degree of longitude shrinks towards the poles
    dy = (lat2 - lat1) * meters_per_deg
    dx = (lon2 - lon1) * meters_per_deg * np.cos(mean_lat)
    return np.sqrt(dx**2 + dy**2)               # flat-earth approximation, exact enough below ~1 km


def cable_length_mst(coords):                   # total length of the minimum spanning tree over the coordinates [m]
    n = len(coords)
    in_tree = [0]                               # start the tree from the first lodge
    total_length = 0.0
    while len(in_tree) < n:                     # keep adding lodges until all are connected
        best_d = None
        best_j = None
        for i in in_tree:                       # every lodge already connected...
            for j in range(n):                  # ...find the nearest lodge not yet connected
                if j not in in_tree:
                    d = distance_m(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                    if best_d is None or d < best_d:
                        best_d = d
                        best_j = j
        total_length += best_d                  # connect it with the shortest possible cable
        in_tree.append(best_j)
    return total_length


def cable_length_routed(coords):                # MST length with a routing uplift for vertical runs and obstacles [m]
    return cable_length_mst(coords) * Cable_routing_factor   # this routed length is what feeds the cabling cost


def cable_mst_edges(coords):                    # same MST as above, but return the connected pairs so the map can draw them
    n = len(coords)
    in_tree = [0]
    edges = []                                  # one (i, j, length) per cable link
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


def coords_to_local_m(coords):                  # convert lat/lon to local east/north metres from the first lodge
    lat0, lon0 = coords[0]                       # use the first lodge as the origin (0, 0)
    meters_per_deg = 111320
    cos_lat = np.cos(np.radians(lat0))
    xs = [(lon - lon0) * meters_per_deg * cos_lat for lat, lon in coords]   # east offset [m]
    ys = [(lat - lat0) * meters_per_deg for lat, lon in coords]             # north offset [m]
    return xs, ys


def plot_cable_network(coords):                 # schematic of the lodges and the interconnection cabling
    xs, ys = coords_to_local_m(coords)
    edges = cable_mst_edges(coords)

    fig, ax = plt.subplots(figsize=(6, 6))

    for i, j, d in edges:                        # draw each cable link and label it with its straight-line length
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="tab:blue", linewidth=2, zorder=1)
        mx = (xs[i] + xs[j]) / 2                  # midpoint of the link, where the length label sits
        my = (ys[i] + ys[j]) / 2
        ax.annotate(f"{d:.1f} m", (mx, my), textcoords="offset points",
                    xytext=(0, 6), ha="center", color="tab:blue")

    ax.scatter(xs, ys, color="tab:red", s=80, zorder=2)   # draw the lodges on top of the cables
    for k in range(len(coords)):
        ax.annotate(f"Lodge {k+1}", (xs[k], ys[k]), textcoords="offset points", xytext=(8, 8))

    total = cable_length_mst(coords)             # straight-line geometry
    routed = cable_length_routed(coords)         # routed length actually used for cost
    ax.set_title(f"Tinos Ecolodge interconnection\n"
                 f"straight-line MST {total:.1f} m, routed {routed:.1f} m (factor {Cable_routing_factor:.2f})")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal")                       # equal scale on both axes so the geometry is not distorted
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    total = cable_length_mst(Tinos_coordinates)
    routed = cable_length_routed(Tinos_coordinates)
    print(f"Straight-line MST length: {total:.1f} m")
    print(f"Routed length (factor {Cable_routing_factor:.2f}): {routed:.1f} m")
    plot_cable_network(Tinos_coordinates)