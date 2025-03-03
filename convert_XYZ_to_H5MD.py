import os
import argparse
import numpy as np
import h5py

def convert_xyz_to_h5md(xyz_file, h5md_file, timestep):
    """
    Convert an XYZ trajectory file to an H5MD file.

    Parameters:
        xyz_file (str): Path to the input XYZ file.
        h5md_file (str): Path to the output H5MD file.
        timestep (float): Timestep between frames in femtoseconds.
    
    The function reads the XYZ file frame by frame, extracts the positions and atomic
    element symbols, computes the time for each frame, and writes the data to an H5MD file.
    """
    positions = []              # List to hold positions for all frames
    times = []                  # List to hold computed time for each frame
    elemental_symbols = []      # List to store atomic element symbols (from the first frame)
    frame_index = 0

    with open(xyz_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break  # End of file reached

            try:
                n_atoms = int(line.strip())
            except ValueError:
                raise ValueError("XYZ file format error: Expected an integer for number of atoms.")

            comment = f.readline().strip()  # Read (and optionally use) the comment line

            frame_positions = []
            frame_symbols = []
            for i in range(n_atoms):
                parts = f.readline().split()
                if len(parts) < 4:
                    raise ValueError("XYZ file format error: Each atom line must have at least 4 entries.")
                # Save the element symbol and the corresponding x, y, z coordinates
                frame_symbols.append(parts[0])
                frame_positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
            positions.append(frame_positions)
            # For the first frame, store the element symbols (assuming ordering remains constant)
            if frame_index == 0:
                elemental_symbols = frame_symbols

            # Compute time for the frame: frame index * timestep (in fs)
            times.append(frame_index * timestep)
            frame_index += 1

    # Convert lists to NumPy arrays for efficient storage
    positions = np.array(positions)  # Shape: (n_frames, n_atoms, 3)
    times = np.array(times)          # Shape: (n_frames,)

    # Create the H5MD file structure
    with h5py.File(h5md_file, 'w') as h5md:
        # Set a minimal H5MD version attribute
        h5md.attrs['h5md_version'] = '1.1'
        
        # Create a group for particle data
        atoms_group = h5md.create_group("atoms")
        
        # Save the element symbols as a dataset (assuming constant elements across frames)
        dt = h5py.string_dtype(encoding='utf-8')
        atoms_group.create_dataset("element", data=np.array(elemental_symbols, dtype=object), dtype=dt)
        
        # Save the positions dataset with compression
        atoms_group.create_dataset("position", data=positions, compression="gzip")
        
        # Create a group for step data and store both frame indices and actual times
        step_group = h5md.create_group("step")
        step_group.create_dataset("step_index", data=5*np.arange(len(times)), compression="gzip")
        step_group.create_dataset("step_time", data=times, compression="gzip")
    
    print("Conversion complete. H5MD file saved as:", h5md_file)

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Convert XYZ trajectory files to H5MD format.")
    parser.add_argument('-t', '--timestep', type=float, default=2.5,
                        help="Timestep between frames in femtoseconds (default: 2.5)")
    
    args = parser.parse_args()
    
    # Process each .xyz file in the input directory "trajectories_xyz"
    for filename in os.listdir("trajectories_xyz"):
        if filename.endswith(".xyz"):
            input_filepath = os.path.join("trajectories_xyz", filename)
            if os.path.isfile(input_filepath):
                base_name = os.path.splitext(filename)[0]
                output_filename = base_name + ".h5md"
                output_filepath = os.path.join("trajectories", output_filename)
                
                # Call the conversion function with the provided timestep argument
                convert_xyz_to_h5md(input_filepath, output_filepath, args.timestep)