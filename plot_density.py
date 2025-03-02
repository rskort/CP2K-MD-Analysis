#!/usr/bin/env python3
import argparse
import logging
import h5py
import numpy as np
import matplotlib.pyplot as plt

def parse_cell(cell_str):
    # Parse a comma-separated string into a tuple of floats.
    try:
        dims = tuple(float(val) for val in cell_str.split(","))
        if len(dims) != 3:
            raise ValueError("Cell dimensions must have three values: A,B,C")
        return dims
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid cell format: {e}")

def main():
    # Set up command-line arguments with one-letter options.
    parser = argparse.ArgumentParser(
        description="Calculate density (atoms/nm³) of a target element vs. distance to a metal surface from h5md trajectories."
    )
    parser.add_argument("filename", help="Input h5md file")
    # Metal element argument (default: Pt)
    parser.add_argument("-m", "--metal", default="Pt", help="Metal element symbol (default: Pt)")
    # Target element argument (default: O)
    parser.add_argument("-t", "--target", default="O", help="Target element symbol (default: O)")
    # Number of layers in the metal surface; default is 4
    parser.add_argument("-l", "--layers", type=int, default=4, help="Number of layers in the metal surface (default: 4)")
    # Skip time in picoseconds; default is 5 ps (step_time is in fs)
    parser.add_argument("-s", "--skip", type=float, default=5, help="Skip first X ps (default: 5 ps)")
    # Number of histogram bins; default 100.
    parser.add_argument("-b", "--bins", type=int, default=100, help="Number of bins for the histogram (default: 100)")
    # Cell dimensions as comma-separated values (default: "14.25,14.81,54.48")
    parser.add_argument("-c", "--cell", type=parse_cell, default="14.25,14.81,54.48",
                        help="Cell dimensions A,B,C (default: 14.25,14.81,54.48)")
    args = parser.parse_args()

    # Set up logging with timestamp.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    logging.info("Opening file: %s", args.filename)
    with h5py.File(args.filename, "r") as f:
        # Read positions and elements. 
        positions = f["atoms/position"]  # shape: (n_steps, n_atoms, 3)
        elements = f["atoms/element"][:]  # assumed constant over time
        if isinstance(elements[0], bytes):
            elements = np.array([el.decode("utf-8") for el in elements])

        # Read the time steps from the file (in fs).
        step_times = f["step/step_time"][:]
        n_steps = positions.shape[0]
        n_atoms = positions.shape[1]
        logging.info("File contains %d steps and %d atoms", n_steps, n_atoms)

        # Convert skip time from ps to fs (1 ps = 1000 fs).
        skip_time_fs = args.skip * 1000

        # Prepare to accumulate histogram counts.
        counts_total = np.zeros(args.bins, dtype=np.float64)
        processed_steps = 0

        # Determine the cross-sectional area from cell dimensions.
        # Assume the provided cell dimensions are in Å. Convert x and y to nm (1 nm = 10 Å).
        cell_A, cell_B, _ = args.cell
        area_nm2 = (cell_A / 10) * (cell_B / 10)

        # List to store all distances if needed (not used for plotting density but for logging/debug)
        all_distances = []

        # Loop over steps.
        for i in range(n_steps):
            time_fs = step_times[i]
            if time_fs < skip_time_fs:
                continue
            processed_steps += 1
            pos = positions[i]  # shape: (n_atoms, 3)

            # Identify metal atoms using the provided metal element.
            metal_mask = (elements == args.metal)
            if not np.any(metal_mask):
                logging.warning("No metal atoms (%s) found at step %d", args.metal, i)
                continue
            metal_z = pos[metal_mask, 2]

            # Determine the number of metal atoms that constitute the top layer.
            n_metal = metal_z.size
            n_top = int(np.ceil(n_metal / args.layers))
            top_metal_z = np.sort(metal_z)[-n_top:]
            surface_z = np.mean(top_metal_z)

            # Identify target element atoms.
            target_mask = (elements == args.target)
            if not np.any(target_mask):
                logging.warning("No target atoms (%s) found at step %d", args.target, i)
                continue
            target_z = pos[target_mask, 2]
            distances = target_z - surface_z
            all_distances.extend(distances.tolist())

            # For each step, update histogram counts. 
            # We use the same bin range for every step.
            step_counts, bin_edges = np.histogram(distances, bins=args.bins)
            counts_total += step_counts

        if processed_steps == 0:
            logging.error("No steps processed after skipping the first %.2f ps", args.skip)
            return

        if counts_total.sum() == 0:
            logging.error("No distances computed. Check if target element %s exists in the dataset.", args.target)
            return

        # Average counts per step.
        avg_counts = counts_total / processed_steps

        # Calculate bin centers.
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Convert bin width from Å to nm for volume calculation.
        bin_widths_nm = (bin_edges[1:] - bin_edges[:-1]) / 10

        # Calculate density in atoms per nm³: density = (avg count) / (area_nm² * bin thickness)
        density = avg_counts / (area_nm2 * bin_widths_nm)

        # Plot the density profile vs. distance.
        plt.figure()
        plt.plot(bin_centers, density, marker='o', linestyle='-')
        plt.xlabel("Distance to surface (Å)")
        plt.ylabel("Density (atoms/nm³)")
        plt.title(f"Density of element {args.target} vs Distance from {args.metal} Surface")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        logging.info("Processed %d steps. Plot displayed successfully.", processed_steps)

if __name__ == "__main__":
    main()
