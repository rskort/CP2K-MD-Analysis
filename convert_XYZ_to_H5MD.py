import os
import argparse
import numpy as np
import h5py
import logging

# Set up logging with time information.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def convert_xyz_to_h5md_fast(xyz_file, h5md_file, md_timestep, cell_dimensions,
                             metal_type, lattice_dimensions):
    """
    Fast conversion from an XYZ file to an H5MD file.
    
    This version reads the entire file into memory and uses vectorized operations
    to parse the coordinate data, eliminating the overhead from per-line processing
    and class-based object creation.
    
    Parameters:
        xyz_file (str): Path to the input XYZ file.
        h5md_file (str): Path to the output H5MD file.
        md_timestep (float): MD timestep between frames in femtoseconds.
        cell_dimensions (tuple or None): Cell dimensions (Lx, Ly, Lz) if provided.
        metal_type (str): Metal type (e.g. "Pt" or "Au").
        lattice_dimensions (tuple or None): Lattice dimensions if provided.
    """
    logging.info("Starting conversion of %s", xyz_file)
    
    # Read all lines at once.
    with open(xyz_file, 'r') as f:
        lines = f.read().splitlines()
    
    pointer = 0
    frame_positions = []
    times = []
    frame_index = 0

    # Read number of atoms from the first frame to assume constant count.
    if pointer < len(lines):
        try:
            n_atoms = int(lines[pointer].strip())
        except ValueError:
            raise ValueError("XYZ file format error: Expected an integer for number of atoms.")
    else:
        raise ValueError("XYZ file is empty.")
    
    # Get elemental symbols from the first frame.
    # Skip the first two lines (atom count and comment) then read n_atoms lines.
    first_frame_symbols = [line.split()[0] for line in lines[pointer+2:pointer+2+n_atoms]]
    
    # Process each frame.
    while pointer < len(lines):
        try:
            n_atoms_frame = int(lines[pointer].strip())
        except ValueError:
            raise ValueError("XYZ file format error: Expected an integer for number of atoms.")
        if n_atoms_frame != n_atoms:
            raise ValueError("Inconsistent number of atoms between frames.")
        pointer += 2  # Skip the atom count and the comment line.
        
        # Grab the next n_atoms lines and parse positions.
        frame_data = lines[pointer:pointer+n_atoms]
        pointer += n_atoms
        
        # Use numpy.fromstring to quickly extract the three coordinates from each line.
        # We use count=4 so that we read the element and the three numbers, then slice off the element.
        frame_array = np.array([np.array(line.split()[1:], dtype=np.float64) for line in frame_data])
        frame_positions.append(frame_array)
        times.append(frame_index * md_timestep)
        frame_index += 1

    positions = np.array(frame_positions)  # Shape: (n_frames, n_atoms, 3)
    times = np.array(times)
    n_frames = positions.shape[0]
    
    # Create H5MD file structure and write data.
    with h5py.File(h5md_file, 'w') as h5md:
        h5md.attrs['h5md_version'] = '1.1'
        h5md.attrs['metal_type'] = metal_type
        if lattice_dimensions is not None:
            h5md.attrs['lattice_dimensions'] = np.array(lattice_dimensions)
        if cell_dimensions:
            h5md.create_dataset("cell_dimensions", data=np.array(cell_dimensions))
        
        # Write atomic data.
        atoms_group = h5md.create_group("atoms")
        dt = h5py.string_dtype(encoding='utf-8')
        atoms_group.create_dataset("element", data=np.array(first_frame_symbols, dtype=object),
                                     dtype=dt)
        atoms_group.create_dataset("position", data=positions, compression="gzip")
        
        # Write time step data.
        step_group = h5md.create_group("step")
        step_group.create_dataset("step_index", data=5 * np.arange(n_frames), compression="gzip")
        step_group.create_dataset("step_time", data=times, compression="gzip")
    
    logging.info("Conversion complete. H5MD file saved as: %s", h5md_file)

# -------------------------
# Main Execution
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Conversion of XYZ trajectory files to H5MD format."
    )
    parser.add_argument('-t', '--timestep', type=float, default=2.5,
                        help="MD timestep between frames in femtoseconds (default: 2.5)")
    parser.add_argument('-c', '--cell', type=float, nargs=3, default=None,
                        help="Cell dimensions: three floats for Lx, Ly, and Lz (default: None)")
    parser.add_argument('-i', '--input', type=str, default="trajectories_xyz",
                        help="Path to the input XYZ file or directory (default: trajectories_xyz)")
    parser.add_argument('-m', '--metal', type=str, default="Pt",
                        help="Metal type of the electrode (e.g. Pt or Au; default: Pt)")
    parser.add_argument('-l', '--lattice', type=int, nargs=3, default=None,
                        help="Lattice dimensions: three integers for X, Y, and Z (layers)")
    args = parser.parse_args()
    
    # Determine input files.
    input_path = args.input
    xyz_files = []
    if os.path.isfile(input_path):
        xyz_files.append(input_path)
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith(".xyz"):
                full_path = os.path.join(input_path, filename)
                if os.path.isfile(full_path):
                    xyz_files.append(full_path)
    else:
        raise ValueError(f"Provided input path '{input_path}' is neither a file nor a directory.")
    
    # Create output directory if it does not exist.
    output_dir = "trajectories"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each XYZ file.
    for xyz_file in xyz_files:
        base_name = os.path.splitext(os.path.basename(xyz_file))[0]
        output_filename = base_name + ".h5md"
        output_filepath = os.path.join(output_dir, output_filename)
        convert_xyz_to_h5md_fast(xyz_file, output_filepath, args.timestep, args.cell,
                                 args.metal, args.lattice)
