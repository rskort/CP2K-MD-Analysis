import time
import numpy as np
import h5py

def load_xyz(xyz_file):
    """
    Load coordinates from an XYZ file.
    Returns a NumPy array of shape (n_frames, n_atoms, 3).
    """
    positions = []
    with open(xyz_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            n_atoms = int(line.strip())
            # Skip comment line
            f.readline()
            frame = []
            for i in range(n_atoms):
                parts = f.readline().split()
                frame.append([float(parts[1]), float(parts[2]), float(parts[3])])
            positions.append(frame)
    return np.array(positions)

def load_h5md(h5md_file):
    """
    Load coordinates from an H5MD file.
    Assumes positions are stored under the group "atoms/position".
    """
    with h5py.File(h5md_file, 'r') as f:
        positions = f['atoms/position'][:]
    return positions

def simple_analysis(positions):
    """
    Compute the center-of-mass (COM) for each frame assuming equal atomic masses.
    Returns an array of COM coordinates with shape (n_frames, 3).
    """
    return np.mean(positions, axis=1)

if __name__ == '__main__':
    xyz_file = "Pt111_Cs1.xyz"
    h5md_file = "Pt111_Cs1.h5md"
    
    # Timing the loading and analysis for XYZ file
    t0 = time.time()
    pos_xyz = load_xyz(xyz_file)
    com_xyz = simple_analysis(pos_xyz)
    t1 = time.time()
    time_xyz = t1 - t0
    
    # Timing the loading and analysis for H5MD file
    t0 = time.time()
    pos_h5md = load_h5md(h5md_file)
    com_h5md = simple_analysis(pos_h5md)
    t1 = time.time()
    time_h5md = t1 - t0
    
    print("XYZ loading and analysis time: {:.4f} s".format(time_xyz))
    print("H5MD loading and analysis time: {:.4f} s".format(time_h5md))
