#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from read_h5md import Simulation
from itertools import cycle

def main():
    parser = argparse.ArgumentParser(
        description="Calculate density (atoms/nm³ or molecules/nm³) vs. distance from a metal surface using a vectorized approach."
    )
    parser.add_argument("filenames", nargs="+", help="Input h5md file(s)")
    parser.add_argument("-s", "--skip", type=float, default=5, help="Skip first X ps (default: 5 ps)")
    parser.add_argument("-b", "--bins", type=int, default=500, help="Number of histogram bins (default: 500)")
    parser.add_argument("-t", "--target", type=str, default="O", help="Target element symbol (default: O). Use 'H2O' for water.")
    # If -i is given, the initial position of the target atom is plotted as a dot on the graph once.
    parser.add_argument("-i", "--initial_position", action="store_true", help="Plot initial position of target atom")
    # If -v is given, debug messages are shown.
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug messages")
    args = parser.parse_args()

    # Set up logging.
    # If -v is given, debug messages are shown. Otherwise, only info messages are shown.
    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    # Color list to cycle through for plotting.
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    plt.figure()
    for file_path in args.filenames:
        data = Simulation(file_path)
        
        # Define bin edges (in Å).
        bin_edges = data.cell_dimensions[2] * np.linspace(0, 1, args.bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        distances_list = []
        valid_count = 0

        # Compute the density of the target element vs. distance from the metal surface.
        if args.target == "H2O":
            # For each frame, compute water COMs with the KD-tree approach.
            for i in range(len(data.trajectory.atoms.positions)):
                if data.trajectory.times[i] < args.skip * 1000:
                    continue
                valid_count += 1

                if data.trajectory.water_coms.size:
                    d = np.abs(data.trajectory.water_coms[i, 2] - data.trajectory.surface_zs[i])
                    distances_list.append(d)
        else:
            # Vectorized approach for a target element.
            target_mask = (data.trajectory.atoms.elements == args.target)
            if not np.any(target_mask):
                logging.warning("No atoms with target element '%s' in file %s", args.target, file_path)
                continue
            target_z = data.trajectory.atoms.positions[:, target_mask, 2]  # shape: (n_frames, n_target)
            for i in range(len(data.trajectory.atoms.positions)):
                if data.trajectory.times[i] < args.skip * 1000:
                    continue
                valid_count += 1
                # Compute distances for all target atoms in frame i.
                d = np.abs(target_z[i] - data.trajectory.surface_zs[i])
                distances_list.append(d)
                            
        if valid_count == 0:
            logging.warning("No valid frames processed for file: %s", file_path)
            continue

        # Concatenate distances from all frames.
        all_distances = np.concatenate(distances_list)
        counts, _ = np.histogram(all_distances, bins=bin_edges)

        # Compute density: convert area from Å² to nm² and bin thickness from Å to nm.
        density = counts / (data.cell_dimensions[0] * data.cell_dimensions[1] * (bin_edges[1] - bin_edges[0]) * valid_count)
        color = next(colors)

        plt.plot(bin_centers, density, linestyle='-', label=data.project_name, color=color)
        plt.fill_between(bin_centers, density, color=color, alpha=0.2)

        # Plot initial position in distance from surface in Å of each target atom if -i is given.
        if args.initial_position:
            if args.target == "H2O":
                initial_positions = data.trajectory.water_coms[0, 2] - data.trajectory.surface_zs[0]
            else:
                initial_positions = target_z[0] - data.trajectory.surface_zs[0]
            # Plot the initial positions as hollow circles in the same color as the density plot.
            plt.plot(initial_positions, np.zeros_like(initial_positions), 'o', color=color, fillstyle='none', linewidth=2)
            
    plt.xlabel("Distance to surface (Å)")
    if args.target == "H2O":
        plt.ylabel("Density of H2O (molecules/nm³)")
    else:
        plt.ylabel(f"Density of {args.target} (atoms/nm³)")
    
    # Set x-axis limits: Start at minimal distance (-1 A) and end at the maximum distance (+1 A) from the surface in all processed files
    min_distance = min([np.min(distances) for distances in distances_list])
    max_distance = max([np.max(distances) for distances in distances_list])
    plt.xlim(min_distance - 1, max_distance + 1)

    plt.title(f"Density of {args.target} vs Distance from {data.metal_type} Surface")
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    
    logging.debug("Plotting data from all processed files.")
    plt.show()

if __name__ == "__main__":
    main()
