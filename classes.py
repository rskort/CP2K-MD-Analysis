#!/usr/bin/env python3
# classes.py
# Define the Simulation class for processing XYZ files.

import os
import pickle
import logging
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

# Configure logging to include time information.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Simulation:
    """
    Processes an XYZ file to extract simulation metadata and trajectory data.
    """
    class Trajectories:
        """
        Container for simulation trajectories.
        """
        class Positions:
            """
            Holds categorized positions for each frame.
            """
            def __init__(self) -> None:
                self.all: Optional[np.ndarray] = None         # Shape: (n_frames, n_atoms, 3)
                self.metal: Optional[np.ndarray] = None       # Metal atom positions.
                self.adsorbates: Optional[np.ndarray] = None  # Adsorbate hydrogen positions.
                self.water: Optional[np.ndarray] = None       # Water molecules' center-of-mass positions.
                self.ions: Optional[np.ndarray] = None        # Ion positions.
                self.watO: Optional[np.ndarray] = None        # Water oxygen positions.
                self.watH: Optional[np.ndarray] = None        # Water hydrogen positions.

        def __init__(self) -> None:
            self.times: Optional[np.ndarray] = None         # 1D array of frame times.
            self.surface_z: Optional[np.ndarray] = None       # Metal surface z-coordinate per frame.
            self.positions: Simulation.Trajectories.Positions = Simulation.Trajectories.Positions()

    def __init__(self,
                 filename: str,
                 md_timestep: float,
                 cell_dimensions: Optional[List[float]],
                 lattice_dimensions: Optional[Tuple[int, int, int]],
                 electrode_potential: float) -> None:
        """
        Initialize a Simulation instance and process the input file.
        """
        self.filename = filename
        self.timestep = md_timestep
        self.cell_dimensions = cell_dimensions
        self.lattice_dimensions = lattice_dimensions
        self.electrode_potential = electrode_potential

        self.project_name = os.path.splitext(os.path.basename(filename))[0]
        self.metal_type: Optional[str] = None  # recognized metal element
        self.ions: Optional[List[str]] = None   # list of ion element symbols

        self.trajectories: Simulation.Trajectories = Simulation.Trajectories()
        self._process_file()

    def _process_file(self) -> None:
        """Read and process the XYZ file to extract metadata and trajectory information."""
        logging.info("Starting processing of %s", self.filename)

        # Read all lines from the file.
        with open(self.filename, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()

        if not lines:
            raise ValueError("XYZ file is empty.")

        # --- Process first frame for atom classification ---
        try:
            n_atoms = int(lines[0].strip())
        except ValueError:
            raise ValueError("XYZ file format error: Expected an integer for number of atoms.")

        # Parse first frame (skip count and comment lines)
        first_frame_lines = lines[2:2 + n_atoms]
        first_frame_symbols = []
        first_frame_positions = np.empty((n_atoms, 3), dtype=np.float64)
        for i, line in enumerate(first_frame_lines):
            parts = line.split()
            first_frame_symbols.append(parts[0])
            first_frame_positions[i] = np.array(parts[1:], dtype=np.float64)
        element_counts = Counter(first_frame_symbols)
        logging.info("Element counts in first frame: %s", element_counts)

        # --- Identify metal atoms ---
        if self.lattice_dimensions is not None:
            expected_metal_atoms = self.lattice_dimensions[0] * self.lattice_dimensions[1] * self.lattice_dimensions[2]
            metal_possibilities = ["Cu", "Ag", "Au", "Pt", "Pd", "Rh", "Ir", "Ni", "Co", "Fe", "Ru", "Os"]
            metal_candidates = [
                el for el, count in element_counts.items()
                if count == expected_metal_atoms and el in metal_possibilities
            ]
            if not metal_candidates:
                recognized_metal = input("No metal element found in the expected number of atoms. "
                                         "Please enter the metal element: ")
            elif len(metal_candidates) == 1:
                recognized_metal = metal_candidates[0]
            else:
                recognized_metal = input(f"Multiple metal candidates found: {metal_candidates}. "
                                         "Please enter the metal element: ")
        else:
            recognized_metal = input("Lattice dimensions not provided. Please enter the metal element: ")
        self.metal_type = recognized_metal
        metal_indices = [i for i, el in enumerate(first_frame_symbols) if el == recognized_metal]
        logging.info("Recognized metal element: %s (count=%d)", recognized_metal, len(metal_indices))

        # --- Identify water molecules and adsorbate hydrogens ---
        oxygen_indices = [i for i, el in enumerate(first_frame_symbols) if el == "O"]
        hydrogen_indices = [i for i, el in enumerate(first_frame_symbols) if el == "H"]
        num_O = len(oxygen_indices)
        num_H_total = len(hydrogen_indices)
        logging.info("Oxygen count: %d, Hydrogen count: %d", num_O, num_H_total)

        expected_adsorbate_count = num_H_total - 2 * num_O if num_H_total > 2 * num_O else 0

        # Compute metal surface z-coordinate from the first frame.
        if metal_indices:
            metal_pos = first_frame_positions[metal_indices, :]
            num_layers = self.lattice_dimensions[2] if self.lattice_dimensions is not None else 1
            num_in_layer = len(metal_indices) // num_layers
            sorted_z = np.sort(metal_pos[:, 2])[::-1]
            metal_surface_z_first = np.mean(sorted_z[:num_in_layer])
            logging.info("Computed metal surface z-coordinate (first frame): %.3f", metal_surface_z_first)
        else:
            metal_surface_z_first = -np.inf
            logging.warning("No metal atoms found; cannot compute metal surface z-coordinate.")

        # Assign adsorbate hydrogens (those closest to the metal surface).
        hydrogen_sorted = sorted(hydrogen_indices, key=lambda i: first_frame_positions[i, 2])
        adsorbate_indices = hydrogen_sorted[:expected_adsorbate_count] if expected_adsorbate_count > 0 else []
        logging.info("Assigned %d hydrogen atoms as adsorbates.", len(adsorbate_indices))

        # The remaining hydrogens are candidate water hydrogens.
        candidate_water_H = set(hydrogen_indices) - set(adsorbate_indices)
        if len(candidate_water_H) != 2 * num_O:
            raise ValueError(f"Expected {2 * num_O} water hydrogens but found {len(candidate_water_H)} "
                             "after adsorbate assignment.")

        # --- Pair each oxygen with its two nearest water hydrogens (vectorized per oxygen) ---
        water_pairs = []  # Each tuple: (oxygen_index, h1_index, h2_index)
        OH_threshold = 1.2  # Ångström
        candidate_water_H = np.array(list(candidate_water_H))
        for o in oxygen_indices:
            diff = first_frame_positions[candidate_water_H] - first_frame_positions[o]  # shape: (n, 3)
            if self.cell_dimensions is not None:
                cell_dims = np.array(self.cell_dimensions)
                diff -= cell_dims * np.round(diff / cell_dims)
            dists = np.linalg.norm(diff, axis=1)
            sorted_indices = np.argsort(dists)
            if len(sorted_indices) < 2:
                raise ValueError(f"Not enough water hydrogens to pair with oxygen at index {o}.")
            if dists[sorted_indices[1]] > OH_threshold:
                logging.warning("For oxygen at index %d, second closest H distance %.2f Å exceeds threshold %.2f.",
                                o, dists[sorted_indices[1]], OH_threshold)
            h1 = candidate_water_H[sorted_indices[0]]
            h2 = candidate_water_H[sorted_indices[1]]
            water_pairs.append((o, int(h1), int(h2)))
            candidate_water_H = candidate_water_H[(candidate_water_H != h1) & (candidate_water_H != h2)]
        water_oxygen_indices = [pair[0] for pair in water_pairs]
        water_H_indices = [pair[1] for pair in water_pairs] + [pair[2] for pair in water_pairs]

        # --- Identify ions ---
        classified_indices = set(metal_indices) | set(oxygen_indices) | set(water_H_indices) | set(adsorbate_indices)
        ion_candidate_indices = [i for i in range(n_atoms) if i not in classified_indices]
        common_ions = ["Li", "Na", "K", "Cs", "Rb", "F", "Cl", "Br", "I"]
        ion_indices = []
        for i in ion_candidate_indices:
            el = first_frame_symbols[i]
            if el in common_ions:
                ion_indices.append(i)
            else:
                answer = input(f"Element '{el}' at index {i} is not recognized as metal, water, or a common ion. "
                               "Should it be classified as an ion? (y/n): ").strip().lower()
                if answer == 'y':
                    ion_indices.append(i)
                else:
                    raise ValueError(f"Element '{el}' at index {i} could not be classified.")
        logging.info("Ion atoms assigned: %d", len(ion_indices))

        all_classified = set(metal_indices) | set(oxygen_indices) | set(water_H_indices) | set(adsorbate_indices) | set(ion_indices)
        if len(all_classified) != n_atoms:
            unclassified = set(range(n_atoms)) - all_classified
            raise ValueError(f"Not all atoms were classified. Unclassified indices: {unclassified}")

        # --- Process all frames ---
        frame_size = n_atoms + 2  # Each frame has a count and a comment line
        n_frames = len(lines) // frame_size
        logging.info("Total frames detected: %d", n_frames)

        # Preallocate arrays for positions, times, and metal surface z.
        positions_all = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
        times_arr = np.empty(n_frames, dtype=np.float64)
        surface_z_arr = np.empty(n_frames, dtype=np.float64)

        # Insert the already-parsed first frame.
        positions_all[0] = first_frame_positions
        times_arr[0] = 0.0
        surface_z_arr[0] = metal_surface_z_first

        # Process remaining frames.
        for frame in range(1, n_frames):
            start = frame * frame_size + 2  # Skip count and comment lines
            end = start + n_atoms
            # Parse each line: split and convert columns 1:4 into float
            try:
                frame_array = np.array(
                    [list(map(float, line.split()[1:4])) for line in lines[start:end]],
                    dtype=np.float64
                )
            except ValueError as e:
                logging.error("Error parsing frame %d: %s", frame, e)
                raise

            positions_all[frame] = frame_array
            times_arr[frame] = frame * self.timestep

            # Compute the metal surface z-coordinate.
            metal_frame = frame_array[metal_indices, :]
            if metal_frame.size > 0:
                num_layers = self.lattice_dimensions[2] if self.lattice_dimensions is not None else 1
                num_in_layer = len(metal_indices) // num_layers
                sorted_z = np.sort(metal_frame[:, 2])[::-1]
                surface_z_arr[frame] = np.mean(sorted_z[:num_in_layer])
            else:
                surface_z_arr[frame] = -np.inf


        # --- Vectorized computation of water molecules' center-of-mass (COM) ---
        water_pairs_arr = np.array(water_pairs, dtype=int)  # Shape: (num_water, 3)
        pos_O_all = positions_all[:, water_pairs_arr[:, 0], :]  # (n_frames, num_water, 3)
        pos_H1_all = positions_all[:, water_pairs_arr[:, 1], :]
        pos_H2_all = positions_all[:, water_pairs_arr[:, 2], :]
        if self.cell_dimensions is not None:
            cell_dims = np.array(self.cell_dimensions).reshape(1, 1, 3)
            delta_H1 = ((pos_H1_all - pos_O_all + cell_dims / 2) % cell_dims) - cell_dims / 2
            delta_H2 = ((pos_H2_all - pos_O_all + cell_dims / 2) % cell_dims) - cell_dims / 2
            pos_H1_corr = pos_O_all + delta_H1
            pos_H2_corr = pos_O_all + delta_H2
        else:
            pos_H1_corr = pos_H1_all
            pos_H2_corr = pos_H2_all
        water_com = (16.0 * pos_O_all + pos_H1_corr + pos_H2_corr) / 18.0

        # --- Populate trajectories ---
        traj_positions = self.trajectories.positions
        traj_positions.all = positions_all
        traj_positions.metal = positions_all[:, metal_indices, :]
        traj_positions.adsorbates = positions_all[:, adsorbate_indices, :] if adsorbate_indices else np.empty((n_frames, 0, 3))
        traj_positions.ions = positions_all[:, ion_indices, :] if ion_indices else np.empty((n_frames, 0, 3))
        traj_positions.watO = positions_all[:, water_oxygen_indices, :]
        traj_positions.watH = positions_all[:, water_H_indices, :]
        traj_positions.water = water_com

        self.trajectories.times = times_arr
        self.trajectories.surface_z = surface_z_arr

        # Save ion element symbols (from first frame) for metadata.
        self.ions = [first_frame_symbols[i] for i in ion_indices]
        logging.info("Processing complete. Total frames processed: %d", n_frames)

    def save(self, output_file: str) -> None:
        """
        Save the Simulation object to a pickle file.
        """
        with open(output_file, 'wb') as pf:
            pickle.dump(self, pf)
        logging.info("Simulation saved to: %s", output_file)

    def __repr__(self) -> str:
        return f"Simulation(project='{self.project_name}', metal='{self.metal_type}')"
