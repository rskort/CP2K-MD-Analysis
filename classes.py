#!/usr/bin/env python3
# classes.py
# Author: R.S. Kort
# Date: March 25, 2025

# Define the Simulation class for processing XYZ files.


import os
import pickle
import logging
from collections import Counter
from typing import List, Optional, Tuple, Set

import numpy as np

# Configure logging to include timestamp, level, and message.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Simulation:
    """
    Processes an XYZ file to extract simulation metadata and trajectory data.
    """

    class Trajectories:
        """
        Container for simulation trajectory data.
        """

        class Positions:
            """
            Stores categorized atomic positions for each simulation frame.
            """
            def __init__(self) -> None:
                # All positions with shape (n_frames, n_atoms, 3)
                self.all: Optional[np.ndarray] = None
                # Metal atom positions.
                self.metal: Optional[np.ndarray] = None
                # Adsorbate hydrogen positions.
                self.adsorbates: Optional[np.ndarray] = None
                # Water molecules' center-of-mass positions.
                self.water: Optional[np.ndarray] = None
                # Ion positions.
                self.ions: Optional[np.ndarray] = None
                # Water oxygen positions.
                self.watO: Optional[np.ndarray] = None
                # Water hydrogen positions.
                self.watH: Optional[np.ndarray] = None

        def __init__(self) -> None:
            # 1D array of frame times.
            self.times: Optional[np.ndarray] = None
            # Metal surface z-coordinate per frame.
            self.surface_z: Optional[np.ndarray] = None
            # Categorized positions.
            self.positions: Simulation.Trajectories.Positions = Simulation.Trajectories.Positions()

    def __init__(self,
                 filename: str,
                 xyz_timestep: float,
                 cell_dimensions: Optional[List[float]],
                 lattice_dimensions: Optional[Tuple[int, int, int]],
                 electrode_potential: float) -> None:
        """
        Initialize a Simulation instance and process the given XYZ file.
        
        Args:
            filename (str): Path to the XYZ file.
            xyz_timestep (float): Timestep between frames in the xyz file containing the MD simulation.
            cell_dimensions (Optional[List[float]]): Simulation cell dimensions.
            lattice_dimensions (Optional[Tuple[int, int, int]]): Lattice dimensions as a tuple.
            electrode_potential (float): Electrode potential value.
        """
        self.filename = filename
        self.timestep = xyz_timestep
        self.cell_dimensions = cell_dimensions
        self.lattice_dimensions = lattice_dimensions
        self.electrode_potential = electrode_potential

        # Derive the project name from the filename.
        self.project_name = os.path.splitext(os.path.basename(filename))[0]
        self.metal_type: Optional[str] = None  # Recognized metal element.
        self.ions: Optional[List[str]] = None    # List of ion element symbols.

        self.trajectories: Simulation.Trajectories = Simulation.Trajectories()
        self._process_file()

    def _process_file(self) -> None:
        """
        Read and process the XYZ file to extract metadata and trajectory information.
        """
        logging.info("Starting processing of file: %s", self.filename)

        # Read all lines from the file.
        with open(self.filename, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()

        if not lines:
            raise ValueError("XYZ file is empty.")

        # Process the first frame to classify atoms.
        try:
            n_atoms = int(lines[0].strip())
        except ValueError:
            raise ValueError("XYZ file format error: Expected an integer for number of atoms on the first line.")

        # Extract the first frame (skip the first two lines: atom count and comment).
        first_frame_lines = lines[2:2 + n_atoms]
        first_frame_symbols = []
        first_frame_positions = np.empty((n_atoms, 3), dtype=np.float64)
        for i, line in enumerate(first_frame_lines):
            parts = line.split()
            first_frame_symbols.append(parts[0])
            first_frame_positions[i] = np.array(parts[1:], dtype=np.float64)

        # Create a Counter object to count occurrences of each element in the first frame.
        element_counts = Counter(first_frame_symbols)
        logging.info("Element counts in first frame: %s", element_counts)


        # --- Identify metal atoms ---
        if self.lattice_dimensions is not None:
            # Calculate the expected number of metal atoms based on lattice dimensions.
            expected_metal_atoms = self.lattice_dimensions[0] * self.lattice_dimensions[1] * self.lattice_dimensions[2]
            metal_possibilities = ["Cu", "Ag", "Au", "Pt", "Pd", "Rh", "Ir", "Ni", "Co", "Fe", "Ru", "Os"]
            # Find elements with the expected number of atoms that are also possible lattice metals.
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

        # Assign the recognized metal element.
        self.metal_type = recognized_metal
        # Find the indices of the recognized metal atoms.
        metal_indices = [i for i, el in enumerate(first_frame_symbols) if el == recognized_metal]
        logging.info("Recognized metal element: %s (count = %d)", recognized_metal, len(metal_indices))


        # --- Identify water molecules and adsorbate hydrogens ---
        oxygen_indices = [i for i, el in enumerate(first_frame_symbols) if el == "O"]
        hydrogen_indices = [i for i, el in enumerate(first_frame_symbols) if el == "H"]
        num_O = len(oxygen_indices)
        num_H_total = len(hydrogen_indices)
        logging.info("Oxygen count: %d, Hydrogen count: %d", num_O, num_H_total)

        # Determine the expected number of adsorbate hydrogens.
        expected_adsorbate_count = num_H_total - 2 * num_O if num_H_total > 2 * num_O else 0

        # Compute the metal surface z-coordinate from the first frame.
        if metal_indices:
            metal_positions = first_frame_positions[metal_indices, :]
            num_layers = self.lattice_dimensions[2] if self.lattice_dimensions is not None else 1
            num_in_layer = len(metal_indices) // num_layers
            # Sort metal atoms by z-coordinate (descending order) and average the top layer.
            sorted_z = np.sort(metal_positions[:, 2])[::-1]
            metal_surface_z_first = np.mean(sorted_z[:num_in_layer])
            logging.info("Computed metal surface z-coordinate (first frame): %.3f", metal_surface_z_first)
        else:
            # Default to 0 if no metal atoms are found.
            metal_surface_z_first = 0
            logging.warning("No metal atoms found; cannot compute metal surface z-coordinate.")

        # Assign adsorbate hydrogens (those closest to the metal surface).
        # Compute the difference from the metal surface for each hydrogen atom and sort by closeness.
        hydrogen_sorted = sorted(hydrogen_indices, key=lambda i: abs(first_frame_positions[i, 2] - metal_surface_z_first))
        adsorbate_indices = hydrogen_sorted[:expected_adsorbate_count] if expected_adsorbate_count > 0 else []
        logging.info("Assigned %d hydrogen atoms as adsorbates.", len(adsorbate_indices))

        # The remaining hydrogens are candidate water hydrogens.
        # We remove the hydrogens already assigned as adsorbates.
        candidate_water_H: Set[int] = set(hydrogen_indices) - set(adsorbate_indices)
        # Check that the remaining candidate water hydrogens match the expected count (2 per oxygen).
        if len(candidate_water_H) != 2 * num_O:
            raise ValueError(f"Expected {2 * num_O} water hydrogens but found {len(candidate_water_H)} "
                            "after adsorbate assignment.")


        # --- Pair each oxygen with its two nearest water hydrogens ---
        # This list will store tuples of (oxygen_index, first_hydrogen_index, second_hydrogen_index)
        water_pairs: List[Tuple[int, int, int]] = []
        OH_threshold: float = 1.2  # Threshold distance in Ångström for the second closest hydrogen

        # Convert the candidate set to a NumPy array for easier numerical operations.
        candidate_water_H = np.array(list(candidate_water_H))

        # Loop over every oxygen atom index to pair it with its nearest water hydrogens.
        for o in oxygen_indices:
            # Calculate the vector differences between the candidate hydrogens' positions and the oxygen's position.
            diff: np.ndarray = first_frame_positions[candidate_water_H] - first_frame_positions[o]
            
            # If periodic boundary conditions are defined, adjust the differences accordingly.
            if self.cell_dimensions is not None:
                cell_dims: np.ndarray = np.array(self.cell_dimensions)
                # Correct for periodic boundaries by subtracting the nearest cell image.
                diff -= cell_dims * np.round(diff / cell_dims)
            
            # Calculate the Euclidean distances from the oxygen to each candidate hydrogen.
            dists: np.ndarray = np.linalg.norm(diff, axis=1)
            
            # Get the indices that would sort the distances array in ascending order.
            sorted_indices: np.ndarray = np.argsort(dists)
            # `sorted_indices` now contains the indices of the candidate hydrogens sorted by distance to O.
            
            # Ensure that there are at least two candidate hydrogens to pair with the current oxygen.
            if len(sorted_indices) < 2:
                raise ValueError(f"Not enough water hydrogens to pair with oxygen at index {o}.")
            
            # Issue a warning if the second closest hydrogen is farther than the threshold.
            if dists[sorted_indices[1]] > OH_threshold:
                logging.warning(f"For oxygen at index {o}, second closest H distance \
                                {dists[sorted_indices[1]]:.2f} Å exceeds threshold {OH_threshold:.2f}.")
            
            # Assign the two closest hydrogens to the oxygen.
            h1 = candidate_water_H[sorted_indices[0]]
            h2 = candidate_water_H[sorted_indices[1]]
            
            # Append the oxygen-hydrogen pairing as a tuple to the water_pairs list.
            water_pairs.append((o, int(h1), int(h2)))
            
            # Remove the paired hydrogens from the candidate set to avoid reusing them.
            candidate_water_H = candidate_water_H[(candidate_water_H != h1) & (candidate_water_H != h2)]

        # Extract lists of indices for oxygens and hydrogens that are part of the water pairs.
        water_oxygen_indices: List[int] = [pair[0] for pair in water_pairs]         # Should be equal to oxygen_indices in case of only H adsorbates.
        water_H_indices: List[int] = [pair[1] for pair in water_pairs] + [pair[2] for pair in water_pairs]


        # --- Identify ions ---
        # Combine indices for atoms already classified as metals, oxygens, water hydrogens, or adsorbate atoms.
        classified_indices = set(metal_indices) | set(oxygen_indices) | set(water_H_indices) | set(adsorbate_indices)

        # Identify indices of atoms that have not been classified in the previous step.
        ion_candidate_indices = [i for i in range(n_atoms) if i not in classified_indices]

        # Define a list of common ions based on their elemental symbols.
        common_ions = ["Li", "Na", "K", "Cs", "Rb", "F", "Cl", "Br", "I", "Mg", "Ca", "Sr", "Ba", "Zn", "Cd", "Hg"]

        # Initialize an empty list to store indices of atoms that will be classified as ions.
        ion_indices = []

        # Iterate over each candidate atom index that has not yet been classified.
        for i in ion_candidate_indices:
            # Retrieve the element symbol from the first frame using the atom index.
            element = first_frame_symbols[i]
            
            # If the element is in the list of common ions, classify it as an ion.
            if element in common_ions:
                ion_indices.append(i)
            else:
                # For elements not automatically recognized as common ions,
                # prompt the user for confirmation whether it should be classified as an ion.
                answer = input(
                    f"Element '{element}' at index {i} is not recognized as metal, water, or a common ion. "
                    "Should it be classified as an ion? (y/n): "
                ).strip().lower()
                
                # If the user confirms, add the atom index to the ion_indices list.
                if answer == 'y':
                    ion_indices.append(i)
                else:
                    # If the user denies classification, raise an error to indicate an unresolved classification.
                    raise ValueError(f"Element '{element}' at index {i} could not be classified.")

        # Log the total number of atoms that have been classified as ions.
        logging.info("Ion atoms assigned: %d", len(ion_indices))

        # Combine all classified atom indices including metals, oxygens, water hydrogens, adsorbate atoms, and ions.
        all_classified = set(metal_indices) | set(oxygen_indices) | set(water_H_indices) | set(adsorbate_indices) | set(ion_indices)

        # Verify that every atom has been classified; if not, raise an error with details on unclassified indices.
        if len(all_classified) != n_atoms:
            unclassified = set(range(n_atoms)) - all_classified
            raise ValueError(f"Not all atoms were classified. Unclassified indices: {unclassified}")


        # --- Process all frames ---
        # Determine the number of lines that constitute one frame:
        # Each frame contains n_atoms position lines plus two header lines (atom count and comment).
        frame_size = n_atoms + 2  
        # Calculate the total number of frames in the input file.
        n_frames = len(lines) // frame_size
        logging.info("Total frames detected: %d", n_frames)

        # Preallocate arrays to store the trajectory data:
        # positions_all will store 3D coordinates for all atoms in each frame.
        positions_all = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
        # times_arr will hold the simulation time for each frame.
        times_arr = np.empty(n_frames, dtype=np.float64)
        # surface_z_arr will record the z-coordinate of the metal surface for each frame.
        surface_z_arr = np.empty(n_frames, dtype=np.float64)

        # Insert data for the first frame, which has been parsed separately.
        positions_all[0] = first_frame_positions
        times_arr[0] = 0.0  # Starting time for the simulation.
        surface_z_arr[0] = metal_surface_z_first

        # Process the remaining frames (frame indices 1 to n_frames-1).
        for frame in range(1, n_frames):
            # Calculate the starting and ending indices for the atom position lines.
            # Skip the two header lines in each frame.
            start: int = frame * frame_size + 2  
            end: int = start + n_atoms
            try:
                # Parse the current frame:
                # For each line corresponding to an atom, split the line and convert columns 1 to 3 to floats.
                frame_array: np.ndarray = np.array(
                    [list(map(float, line.split()[1:4])) for line in lines[start:end]],
                    dtype=np.float64
                )
            except ValueError as e:
                # Log an error if the parsing fails and re-raise the exception.
                logging.error("Error parsing frame %d: %s", frame, e)
                raise
            # Store the parsed positions for the current frame.
            positions_all[frame] = frame_array
            # Compute and store the simulation time for the current frame.
            times_arr[frame] = frame * self.timestep

            # Compute the z-coordinate of the metal surface for the current frame:
            # Select only the metal atoms from the current frame using their indices.
            metal_frame: np.ndarray = frame_array[metal_indices, :]
            if metal_frame.size > 0:
                # Determine the number of layers in the metal surface.
                # If lattice dimensions are defined, use the third dimension; otherwise, default to one layer.
                num_layers = self.lattice_dimensions[2] if self.lattice_dimensions is not None else 1
                # Estimate the number of atoms in one layer.
                num_in_layer = len(metal_indices) // num_layers
                # Sort the z-coordinates in descending order and take the top values corresponding to one layer.
                sorted_z = np.sort(metal_frame[:, 2])[::-1]
                # Average the top z-values to approximate the surface level.
                surface_z_arr[frame] = np.mean(sorted_z[:num_in_layer])
            else:
                # If no metal atoms are present, assign negative infinity.
                surface_z_arr[frame] = -np.inf

        # --- Compute water molecules' center-of-mass (COM) ---
        # Convert water molecule indices to a numpy array; each water molecule is represented by a triplet of indices.
        water_pairs_arr = np.array(water_pairs, dtype=int)  # Shape: (num_water, 3)
        # Extract the oxygen positions for all water molecules.
        pos_O_all = positions_all[:, water_pairs_arr[:, 0], :]  # Dimensions: (n_frames, num_water, 3)
        # Extract hydrogen positions for all water molecules.
        pos_H1_all = positions_all[:, water_pairs_arr[:, 1], :]
        pos_H2_all = positions_all[:, water_pairs_arr[:, 2], :]
        if self.cell_dimensions is not None:
            # Reshape cell dimensions to allow broadcasting during periodic boundary condition adjustments.
            cell_dims = np.array(self.cell_dimensions).reshape(1, 1, 3)
            # Adjust hydrogen positions to account for periodic boundaries:
            # Compute the minimal image displacement between hydrogen and oxygen positions.
            delta_H1 = ((pos_H1_all - pos_O_all + cell_dims / 2) % cell_dims) - cell_dims / 2
            delta_H2 = ((pos_H2_all - pos_O_all + cell_dims / 2) % cell_dims) - cell_dims / 2
            # Correct the hydrogen positions using the computed displacements.
            pos_H1_corr = pos_O_all + delta_H1
            pos_H2_corr = pos_O_all + delta_H2
        else:
            # If no cell dimensions are provided, use the original hydrogen positions.
            pos_H1_corr = pos_H1_all
            pos_H2_corr = pos_H2_all
        # Calculate the center-of-mass for water molecules using mass-weighted positions:
        # Oxygen mass is approximately 15.999, and each hydrogen mass is approximately 1.00784.
        water_com = (15.999 * pos_O_all + 1.00784 * pos_H1_corr + 1.00784 * pos_H2_corr) / (15.999 + 2 * 1.00784)

        # --- Populate trajectories ---
        # Assign the computed positions and derived quantities to the trajectory data structure.
        traj_positions = self.trajectories.positions
        traj_positions.all = positions_all
        # Extract and store metal atom positions.
        traj_positions.metal = positions_all[:, metal_indices, :]
        # Extract and store adsorbate positions if indices are provided; otherwise, assign an empty array.
        traj_positions.adsorbates = positions_all[:, adsorbate_indices, :] if adsorbate_indices else np.empty((n_frames, 0, 3))
        # Extract and store ion positions if indices are provided; otherwise, assign an empty array.
        traj_positions.ions = positions_all[:, ion_indices, :] if ion_indices else np.empty((n_frames, 0, 3))
        # Store water oxygen and hydrogen positions separately.
        traj_positions.watO = positions_all[:, water_oxygen_indices, :]
        traj_positions.watH = positions_all[:, water_H_indices, :]
        # Store the computed center-of-mass for water molecules.
        traj_positions.water = water_com

        # Store additional trajectory metadata: simulation times and metal surface z-coordinates.
        self.trajectories.times = times_arr
        self.trajectories.surface_z = surface_z_arr

        # Save the ion element symbols from the first frame as metadata for later use.
        self.ions = [first_frame_symbols[i] for i in ion_indices]
        logging.info("Processing complete. Total frames processed: %d", n_frames)


    def save(self, output_file: str) -> None:
        """
        Save the Simulation object to a pickle file.
        
        Args:
            output_file (str): Path to the output pickle file.
        """
        with open(output_file, 'wb') as pf:
            pickle.dump(self, pf)
        logging.info("Simulation saved to: %s", output_file)

    def __repr__(self) -> str:
        return f"Simulation(project='{self.project_name}')"
