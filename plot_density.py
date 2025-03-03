#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from read_h5md import read_h5md
from scipy.spatial import cKDTree  # for water COM search (if needed)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def compute_surface_z(positions, elements, metal_type, lattice_dimensions):
    """
    Compute the surface z for each frame using vectorized operations.
    For each frame, select the metal atoms (where element == metal_type),
    then average the top n_top (n_total // electrode_layers) z values.
    """
    n_frames = positions.shape[0]
    electrode_layers = lattice_dimensions[2] if lattice_dimensions is not None else 1
    metal_mask = (elements == metal_type)
    # If no metal atoms are present, use zero.
    if not np.any(metal_mask):
        return np.zeros(n_frames)
    metal_z = positions[:, metal_mask, 2]  # shape: (n_frames, n_metal)
    surface_z = np.empty(n_frames)
    for i in range(n_frames):
        frame_metal = metal_z[i]
        n_top = frame_metal.size // electrode_layers
        if n_top < 1:
            n_top = frame_metal.size
        # Sort descending and average top n_top values.
        top_vals = np.partition(frame_metal, -n_top)[-n_top:]
        surface_z[i] = np.mean(top_vals)
    return surface_z

def compute_water_coms(positions, elements, cutoff=1.2):
    """
    Compute water centers-of-mass for a single frame using a fast KD-tree.
    This function is applied per frame.
    Water molecules are assumed to consist of one 'O' and two nearest 'H' atoms within the cutoff.
    Returns an array of COM positions.
    """
    # Separate oxygen and hydrogen indices.
    O_idx = np.where(elements == 'O')[0]
    H_idx = np.where(elements == 'H')[0]
    if O_idx.size == 0 or H_idx.size == 0:
        return np.empty((0,3))
    pos_O = positions[O_idx]  # shape: (n_O, 3)
    pos_H = positions[H_idx]  # shape: (n_H, 3)
    tree = cKDTree(pos_H)
    coms = []
    mass_O, mass_H = 16.0, 1.0
    for o, O_pos in zip(O_idx, pos_O):
        # Query all H atoms within cutoff.
        idxs = tree.query_ball_point(O_pos, cutoff)
        if len(idxs) < 2:
            continue
        # Get the two closest H atoms.
        H_candidates = pos_H[idxs]
        distances = np.linalg.norm(H_candidates - O_pos, axis=1)
        two_closest = H_candidates[np.argsort(distances)[:2]]
        # Compute center-of-mass.
        com = (mass_O * O_pos + mass_H * two_closest[0] + mass_H * two_closest[1]) / (mass_O + 2*mass_H)
        coms.append(com)
    if coms:
        return np.array(coms)
    else:
        return np.empty((0,3))

def main():
    parser = argparse.ArgumentParser(
        description="Calculate density (atoms/nm³ or molecules/nm³) vs. distance from a metal surface using a vectorized approach."
    )
    parser.add_argument("filenames", nargs="+", help="Input h5md file(s)")
    parser.add_argument("-s", "--skip", type=float, default=5, help="Skip first X ps (default: 5 ps)")
    parser.add_argument("-b", "--bins", type=int, default=100, help="Number of histogram bins (default: 100)")
    parser.add_argument("-t", "--target", type=str, default="O", help="Target element symbol (default: O). Use 'H2O' for water.")
    args = parser.parse_args()

    plt.figure()
    for file_path in args.filenames:
        sim = read_h5md(file_path)
        positions = sim["positions"]    # shape: (n_frames, n_atoms, 3)
        elements = sim["elements"]      # shape: (n_atoms,)
        step_times = sim["step_times"]  # in fs
        metal_type = sim["metal_type"]
        lattice_dimensions = sim["lattice_dimensions"]
        cell_dimensions = sim["cell_dimensions"]
        project_name = sim["project_name"]
        n_frames = positions.shape[0]

        # Precompute surface_z for all frames.
        surface_z_all = compute_surface_z(positions, elements, metal_type, lattice_dimensions)

        # Define bin edges (in Å).
        bin_edges = cell_dimensions[2] * np.linspace(0, 1, args.bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        distances_list = []
        valid_count = 0

        if args.target == "H2O":
            # For each frame, compute water COMs with the KD-tree approach.
            for i in range(n_frames):
                if step_times[i] < args.skip * 1000:
                    continue
                valid_count += 1
                frame_positions = positions[i]  # shape: (n_atoms, 3)
                # Compute water center-of-mass for the frame.
                water_coms = compute_water_coms(frame_positions, elements)
                if water_coms.size:
                    d = np.abs(water_coms[:, 2] - surface_z_all[i])
                    distances_list.append(d)
        else:
            # Vectorized approach for a target element.
            target_mask = (elements == args.target)
            if not np.any(target_mask):
                logging.warning("No atoms with target element '%s' in file %s", args.target, file_path)
                continue
            target_z = positions[:, target_mask, 2]  # shape: (n_frames, n_target)
            for i in range(n_frames):
                if step_times[i] < args.skip * 1000:
                    continue
                valid_count += 1
                # Compute distances for all target atoms in frame i.
                d = np.abs(target_z[i] - surface_z_all[i])
                distances_list.append(d)
                
        if valid_count == 0:
            logging.warning("No valid frames processed for file: %s", file_path)
            continue

        # Concatenate distances from all frames.
        all_distances = np.concatenate(distances_list)
        counts, _ = np.histogram(all_distances, bins=bin_edges)

        # Compute density: convert area from Å² to nm² and bin thickness from Å to nm.
        area_nm2 = (cell_dimensions[0] * cell_dimensions[1]) / 100.0
        bin_thickness_nm = (bin_edges[1] - bin_edges[0]) / 10.0
        bin_volume_nm3 = area_nm2 * bin_thickness_nm
        density = counts / (valid_count * bin_volume_nm3)

        plt.plot(bin_centers, density, linestyle='-', label=project_name)

    plt.xlabel("Distance to surface (Å)")
    plt.xlim(left=0)
    if args.target == "H2O":
        plt.ylabel("Density of H2O (molecules/nm³)")
    else:
        plt.ylabel(f"Density of {args.target} (atoms/nm³)")
    plt.title(f"Density of {args.target} vs Distance from {metal_type} Surface")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    logging.info("Plotting data from all processed files.")
    plt.show()

if __name__ == "__main__":
    main()
