#!/usr/bin/env python3
import os
import h5py
import numpy as np
import logging

def read_h5md(file_path):
    """
    Read an H5MD file and return a dictionary containing:
      - positions: NumPy array of shape (n_frames, n_atoms, 3)
      - elements: NumPy array of shape (n_atoms,) (assumed constant over frames)
      - step_times: NumPy array of shape (n_frames,)
      - metal_type, cell_dimensions, lattice_dimensions, project_name
    """
    with h5py.File(file_path, 'r') as f:
        # Global attributes.
        metal_type = f.attrs.get('metal_type', None)
        if metal_type is not None and isinstance(metal_type, bytes):
            metal_type = metal_type.decode('utf-8')
        lattice_dimensions = f.attrs.get('lattice_dimensions', None)
        if lattice_dimensions is not None:
            lattice_dimensions = tuple(lattice_dimensions.tolist())
        cell_dimensions = f["cell_dimensions"][()] if "cell_dimensions" in f else None

        # Atoms group.
        atoms_grp = f["atoms"]
        elements = atoms_grp["element"][()]
        if elements.size > 0 and isinstance(elements[0], bytes):
            elements = np.array([el.decode('utf-8') for el in elements])
        positions = atoms_grp["position"][()]  # shape: (n_frames, n_atoms, 3)

        # Step group.
        step_grp = f["step"]
        step_times = step_grp["step_time"][()]  # shape: (n_frames,)
    
    project_name = os.path.splitext(os.path.basename(file_path))[0]
    logging.debug("Reader: %d frames with %d atoms per frame read from %s",
                 positions.shape[0], positions.shape[1], file_path)
    return {
        "positions": positions,
        "elements": elements,
        "step_times": step_times,
        "metal_type": metal_type,
        "lattice_dimensions": lattice_dimensions,
        "cell_dimensions": cell_dimensions,
        "project_name": project_name
    }
