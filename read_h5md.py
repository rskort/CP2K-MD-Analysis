#!/usr/bin/env python3
import os
import h5py
import numpy as np
from functools import cached_property

class Simulation:
    """
    Main class to load and represent an H5MD file structure.
    This class also stores simulation-level metadata.
    """
    def __init__(self, h5md_file_path):
        if not os.path.isfile(h5md_file_path):
            raise FileNotFoundError(f"H5MD file '{h5md_file_path}' does not exist.")
        with h5py.File(h5md_file_path, 'r') as f:
            # Load time-dependent trajectory data from the 'trajectory' group.
            self.trajectory = Trajectory(f["trajectory"], simulation=self)

            # Read attributes from the simulation group.
            self.metal_type = f["simulation"].attrs.get("metal_type")
            self.lattice_dimensions = f["simulation"].attrs.get("lattice_dimensions")
            if self.lattice_dimensions is not None:
                self.lattice_dimensions = tuple(self.lattice_dimensions)
            self.cell_dimensions = f["simulation"]["cell_dimensions"][()] if "cell_dimensions" in f["simulation"] else None
            self.project_name = f["simulation"].attrs.get("project_name")
            self.ion_types = f["simulation"].attrs.get("ion_types")

class Trajectory:
    """Class to store time-dependent simulation data, including atomic information and times."""
    def __init__(self, trajectory_group, simulation):
        self.simulation = simulation
        self.atoms = Atoms(trajectory_group["atoms"])
        self.times = trajectory_group["times"][()]
    
    @property
    def positions(self):
        """
        Shortcut to access atomic positions from the trajectory.
        """
        return self.atoms.positions
    
    @cached_property
    def surface_zs(self):
        """
        Compute the z-coordinate of the surface for each frame and save to self.surface_zs.
        The surface z is computed by averaging the z-values of the top layer of the metal.
        The top metal layer can be found by dividing the total number of metal atoms by the number of layers.
        """
        n_frames = self.positions.shape[0]
        electrode_layers = self.simulation.lattice_dimensions[2] if self.simulation.lattice_dimensions is not None else 1
        metal_mask = (self.atoms.elements == self.simulation.metal_type)
        # If no metal atoms are present, use zeros.
        if not np.any(metal_mask):
            return np.zeros(n_frames)
        metal_z = self.positions[:, metal_mask, 2]  # shape: (n_frames, n_metal)
        surface_z = np.empty(n_frames)
        for i in range(n_frames):
            frame_metal = metal_z[i]
            n_top = frame_metal.size // electrode_layers
            if n_top < 1:
                n_top = frame_metal.size
            # Sort descending and average top n_top values.
            top_vals = np.partition(frame_metal, -n_top)[-n_top:]
            surface_z[i] = np.mean(top_vals)
        self.surface_zs = surface_z
        return self.surface_zs
    
    @cached_property
    def water_coms(self):
        """
        Compute the center-of-mass for water molecules in each frame.
        The COM is determined by accounting for the mass of the O and H atoms in each water molecule.
        """
        water_mask = (self.atoms.elements == "O") | (self.atoms.elements == "H")
        if not np.any(water_mask):
            return np.array([])
        water_positions = self.positions[:, water_mask]
        water_elements = self.atoms.elements[water_mask]
        water_coms = np.empty((len(water_positions), 3))
        for i, (frame_positions, frame_elements) in enumerate(zip(water_positions, water_elements)):
            o_mask = frame_elements == "O"
            h_mask = frame_elements == "H"
            o_mass = 16.0
            h_mass = 1.0
            total_mass = o_mass + 2 * h_mass
            o_position = frame_positions[o_mask]
            h_positions = frame_positions[h_mask]
            com = (o_mass * o_position + h_mass * h_positions.sum(axis=0)) / total_mass
            water_coms[i] = com
        self.water_coms = water_coms
        return self.water_coms
    

class Atoms:
    """Class to store atomic data within simulation frames."""
    def __init__(self, atoms_group):
        # Load the element symbols.
        self.elements = atoms_group["elements"][()]
        if self.elements.size > 0 and isinstance(self.elements[0], bytes):
            # Decode byte strings to UTF-8.
            self.elements = np.array([el.decode('utf-8') for el in self.elements])
        # Load positions: shape (n_frames, n_atoms, 3)
        self.positions = atoms_group["positions"][()]

if __name__ == '__main__':
    # Example usage:
    h5md_file = "trajectories/example.h5md"
    data = Simulation(h5md_file)
    
    # Access simulation-level metadata.
    print("Simulation Metadata:\n----------------")
    print("Project Name:", data.project_name)
    print("Metal Type:", data.metal_type)
    print("Lattice Dimensions:", data.lattice_dimensions)
    print("Cell Dimensions:", data.cell_dimensions)
    print("Ion types:", data.ion_types)
    
    # Access trajectory-level data.
    print("\nTrajectory Data:\n----------------")
    print("Times:", data.trajectory.times)
    print("Element Symbols:", data.trajectory.atoms.elements)
    print("Positions (first atom):", data.trajectory.atoms.positions[0, 0])
    print("Surface Z values:", data.trajectory.surface_zs)
    
    # Access positions using the property for convenience.
    frame_idx = 5
    atom_idx = 0
    position = data.trajectory.positions[frame_idx, atom_idx]
    print(f"Position at frame {frame_idx}, atom {atom_idx}:", position)
