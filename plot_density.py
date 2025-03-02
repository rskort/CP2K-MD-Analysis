#!/usr/bin/env python3
import argparse
import glob
import logging
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_cell(cell_str):
    """Parse a comma-separated string into a tuple of floats."""
    try:
        dims = tuple(float(val) for val in cell_str.split(","))
        if len(dims) != 3:
            raise ValueError("Cell dimensions must have three values: A,B,C")
        return dims
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid cell format: {e}")

def process_file(filename, args):
    """Process a single file and return bin_centers and density values for plotting."""
    logging.info("Processing file: %s", filename)
    with h5py.File(filename, "r") as f:
        positions = f["atoms/position"][:]  # shape: (n_steps, n_atoms, 3)
        step_times = f["step/step_time"][:]
        elements = f["atoms/element"][:]
        if isinstance(elements[0], bytes):
            elements = np.array([el.decode("utf-8") for el in elements])
    
    n_steps = positions.shape[0]

    # Skip steps based on given skip time in ps
    skip_time_fs = args.skip * 1000
    valid_mask = step_times >= skip_time_fs
    if not np.any(valid_mask):
        logging.error("No steps processed after skipping the first %.2f ps in %s", args.skip, filename)
        return None, None
    valid_positions = positions[valid_mask]  # shape: (n_valid, n_atoms, 3)

    cell_A, cell_B, _ = args.cell
    area_nm2 = (cell_A / 10) * (cell_B / 10)

    # Create boolean masks for metal and target elements.
    metal_mask = (elements == args.metal)
    target_mask = (elements == args.target)
    if not np.any(metal_mask):
        logging.error("No metal atoms (%s) found in %s.", args.metal, filename)
        return None, None
    if not np.any(target_mask):
        logging.error("No target atoms (%s) found in %s.", args.target, filename)
        return None, None

    # Get z-coordinates for metal and target atoms.
    metal_z_all = valid_positions[:, metal_mask, 2]  # shape: (n_valid, n_metal)
    n_metal = metal_z_all.shape[1]
    top_n = int(np.ceil(n_metal / args.layers))
    sorted_metal_z = np.sort(metal_z_all, axis=1)
    top_metal_z = sorted_metal_z[:, -top_n:]
    surface_z = np.mean(top_metal_z, axis=1)  # shape: (n_valid,)

    target_z_all = valid_positions[:, target_mask, 2]  # shape: (n_valid, n_target)

    # Instead of computing distances and letting np.histogram choose the bin range,
    # we histogram the absolute target z coordinates using fixed bins that span the full simulation cell.
    target_z_flat = target_z_all.flatten()
    # Create fixed bin edges spanning the full z coordinate of the simulation cell (in Å)
    bin_edges = np.linspace(0, args.cell[2], args.bins + 1)
    counts_total, _ = np.histogram(target_z_flat, bins=bin_edges)

    # To keep the "distance to surface" x-values, shift the bin centers by the average surface position.
    avg_surface_z = np.mean(surface_z)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) - avg_surface_z

    # Compute average counts per valid step.
    avg_counts = counts_total / valid_positions.shape[0]
    # Convert bin widths from Å to nm.
    bin_widths_nm = (bin_edges[1] - bin_edges[0]) / 10
    density = avg_counts / (area_nm2 * bin_widths_nm)

    
    logging.info("Processed %d valid steps in %s", valid_positions.shape[0], filename)
    return bin_centers, density

def main():
    parser = argparse.ArgumentParser(
        description="Calculate density (atoms/nm³) of a target element vs. distance to a metal surface from h5md trajectories."
    )
    # Allow multiple files to be provided.
    parser.add_argument("filename", nargs="+", help="Input h5md file(s) (wildcards allowed)")
    parser.add_argument("-m", "--metal", default="Pt", help="Metal element symbol (default: Pt)")
    parser.add_argument("-t", "--target", default="O", help="Target element symbol (default: O)")
    parser.add_argument("-l", "--layers", type=int, default=4, help="Number of layers in the metal surface (default: 4)")
    parser.add_argument("-s", "--skip", type=float, default=5, help="Skip first X ps (default: 5 ps)")
    parser.add_argument("-b", "--bins", type=int, default=100, help="Number of bins for the histogram (default: 100)")
    parser.add_argument("-c", "--cell", type=parse_cell, default="14.25,14.81,54.48",
                        help="Cell dimensions A,B,C (default: 14.25,14.81,54.48)")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    
    # Expand wildcard patterns into actual file paths
    file_list = []
    for pattern in args.filename:
        file_list.extend(glob.glob(pattern))

    if not file_list:
        logging.error("No files found matching the given pattern(s).")
        return

    plt.figure()
    # Iterate over each resolved file path and process it.
    for file in file_list:
        bin_centers, density = process_file(file, args)
        if bin_centers is None or density is None:
            logging.warning("Skipping file %s due to errors.", file)
            continue
        # Remove extension for the legend.
        label = os.path.splitext(os.path.basename(file))[0]
        plt.plot(bin_centers, density, linestyle='-', label=label)
    
    plt.xlabel("Distance to surface (Å)")
    plt.ylabel("Density (atoms/nm³)")
    plt.title(f"Density of element {args.target} vs Distance from {args.metal} Surface")
    plt.grid(True)
    plt.legend()  # Add legend with file names.
    plt.tight_layout()
    
    logging.info("Plotting data from all processed files.")
    plt.show()

if __name__ == "__main__":
    main()
