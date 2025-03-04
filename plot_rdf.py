#!/usr/bin/env python3
"""
Script to plot the normalized, average radial distribution function (RDF) of a reference element 
to one or more target elements from one or more h5md files. The RDF is computed on a per-frame basis 
(after a specified skip time), normalized, and then averaged.
"""

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from read_h5md import Simulation


def compute_rdf(data, target_symbol, reference_symbol, skip_time, bin_edges):
    """
    Computes the average, normalized RDF for a given simulation data instance.
    
    Normalization is performed per frame using:
      g(r) = (histogram_count) / (number_of_reference_atoms * density_target * shell_volume)
    
    If target and reference are identical, self-interaction distances are excluded.
    
    Parameters:
        data (Simulation): Simulation object loaded from an h5md file.
        target_symbol (str): The chemical symbol for the target atoms.
        reference_symbol (str): The chemical symbol for the reference atoms.
        skip_time (float): Time in ps to skip initial frames.
        bin_edges (np.ndarray): The edges of the histogram bins.
        
    Returns:
        tuple: (bin_centers, rdf_avg) where rdf_avg is the average normalized RDF.
    """
    # Create masks for target and reference atoms.
    target_mask = (data.trajectory.atoms.elements == target_symbol)
    reference_mask = (data.trajectory.atoms.elements == reference_symbol)
    
    if not np.any(target_mask):
        logging.warning("No atoms with target element '%s' found.", target_symbol)
        return None, None
    if not np.any(reference_mask):
        logging.warning("No atoms with reference element '%s' found.", reference_symbol)
        return None, None

    # Get simulation box dimensions.
    box = data.cell_dimensions
    if box is None:
        logging.warning("No cell dimensions found in file.")
        return None, None
    if box.ndim == 1:
        volume = np.prod(box)
    else:
        # Assume box vectors are along the diagonal.
        volume = np.prod(np.diag(box))
    
    # Pre-calculate the shell volumes for each bin.
    shell_volumes = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    
    rdf_list = []
    valid_frames = 0
    
    for i, t in enumerate(data.trajectory.times):
        # Skip frames before the given skip time (convert ps to fs)
        if t < skip_time * 1000:
            continue
        valid_frames += 1
        
        # Get positions for target and reference atoms in the current frame.
        target_positions = data.trajectory.atoms.positions[i, target_mask]
        reference_positions = data.trajectory.atoms.positions[i, reference_mask]
        
        # Compute all pairwise distances between target and reference atoms.
        distances = np.linalg.norm(target_positions[:, np.newaxis] - reference_positions, axis=-1).ravel()
        
        # If target and reference are the same, remove self-interaction distances.
        if target_symbol == reference_symbol:
            distances = distances[distances > 1e-6]
        
        # Histogram the distances.
        hist, _ = np.histogram(distances, bins=bin_edges)
        
        # Determine densities and expected counts:
        num_target = target_positions.shape[0]
        num_reference = reference_positions.shape[0]
        density_target = num_target / volume  # Number density of target atoms
        
        # Expected counts in each bin per reference atom.
        expected_counts = density_target * shell_volumes
        
        # Total expected counts in each bin (accounting for all reference atoms).
        expected_total = num_reference * expected_counts
        
        # Normalize histogram: this yields the RDF g(r)
        with np.errstate(divide='ignore', invalid='ignore'):
            rdf_frame = hist / expected_total
            rdf_frame[np.isnan(rdf_frame)] = 0
        
        rdf_list.append(rdf_frame)
    
    if valid_frames == 0:
        logging.warning("No frames passed the skip time threshold.")
        return None, None

    rdf_array = np.array(rdf_list)
    rdf_avg = rdf_array.mean(axis=0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    return bin_centers, rdf_avg


def plot_rdfs(rdf_data_list):
    """
    Plots the RDF curves for a list of (bin_centers, rdf_avg) tuples.
    
    Parameters:
        rdf_data_list (list): List of tuples (bin_centers, rdf_avg, label) for each dataset.
        target_symbol (str): The chemical symbol for the target atoms.
        reference_symbol (str): The chemical symbol for the reference atoms.
    """
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    plt.figure()
    
    for bin_centers, rdf_avg, label in rdf_data_list:
        color = next(color_cycle)
        plt.plot(bin_centers, rdf_avg, label=label, color=color)
    
    plt.xlabel("Distance (Å)")
    plt.ylabel("RDF")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and plot the average radial distribution function (RDF) "
                    "from h5md simulation files."
    )
    parser.add_argument("filenames", nargs="+", help="Input h5md file(s)")
    parser.add_argument("-s", "--skip", type=float, default=5, 
                        help="Skip first X ps (default: 5 ps)")
    parser.add_argument("-b", "--bins", type=int, default=50, 
                        help="Number of histogram bins (default: 50)")
    parser.add_argument("-m", "--max_distance", type=float, default=10, 
                        help="Maximum distance in Å (default: 10 Å)")
    parser.add_argument("-t", "--target", type=str, default="Li", 
                        help="Target element symbol (default: Li)")
    parser.add_argument("-r", "--reference", type=str, default="O", 
                        help="Reference element symbol (default: O)")
    parser.add_argument("-n", "--normalize", action="store_true", 
                        help="Normalize the RDF")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Enable debug messages")
    
    args = parser.parse_args()

    # Set up logging based on verbosity.
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=log_level)
    
    # Define bin edges for histogram (in Å).
    bin_edges = np.linspace(0, args.max_distance, args.bins + 1)
    
    rdf_data_list = []
    
    for file_path in args.filenames:
        logging.info("Processing file: %s", file_path)
        try:
            sim_data = Simulation(file_path)
        except Exception as e:
            logging.error("Failed to load file %s: %s", file_path, e)
            continue
        
        bin_centers, rdf_avg = compute_rdf(sim_data, args.target, args.reference, args.skip, bin_edges)
        if bin_centers is None or rdf_avg is None:
            logging.warning("Skipping file %s due to missing data.", file_path)
            continue
        
        # Normalize RDF if requested.
        if args.normalize:
            total = rdf_avg.sum()
            if total > 0:
                rdf_avg = rdf_avg / total
            else:
                logging.warning("Sum of RDF is zero in file %s; cannot normalize.", file_path)
        
        label = f"{args.target} - {args.reference} ({sim_data.project_name})"
        rdf_data_list.append((bin_centers, rdf_avg, label))
    
    if rdf_data_list:
        plot_rdfs(rdf_data_list)
    else:
        logging.error("No valid RDF data to plot.")


if __name__ == "__main__":
    main()
