from classes import Simulation
import os

def convert_xyz_to_pickle(xyz_file, xyz_timestep, cell_dims, lattice_dims, electrode_potential):
    """Convert an XYZ file to a Simulation object and save it as a pickle file."""
    sim = Simulation(xyz_file, xyz_timestep, cell_dims, lattice_dims, electrode_potential)
    # Save the Simulation object as a pickle file with the same name as the input XYZ file.
    output_file = os.path.splitext(xyz_file)[0] + ".pkl"
    output_path = os.path.join("data", "simulations", os.path.basename(output_file))
    # Check whether the dir exists and create it if not.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sim.save(output_path)

if __name__ == '__main__':
    xyz_timestep = 2.5 # fs
    cell_dims = (14.25, 14.81, 54.48) # Å
    lattice_dims = (5, 6, 4)
    
    
    """sim_ids =  [
                "Cs1", "Cs2", "Cs3", "Cs4",
                "K1", "K2", "K3", "K4",
                "Li1", "Li2", "Li3", "Li4",
                "Na1", "Na2", "Na3", "Na4",
                "Cs1_H", "Cs2_H", "Cs3_H", "Cs4_H",
                "K1_H", "K2_H", "K3_H", "K4_H",
                "Li1_H", "Li2_H", "Li3_H", "Li4_H",
                "Na1_H", "Na2_H", "Na3_H", "Na4_H",
                "NoIon", "NoIon_H"
                ]
    
    potentials = [
                  -0.12, -0.26, -0.64, -1.16, 
                  -0.22, -0.24, -0.52, -1.14, 
                  -0.13, -0.29, -0.45, -0.86, 
                  -0.25, -0.21, -0.47, -1.04, 
                  -0.22, -0.52, -1.52, -2.25, 
                  -0.08, -0.64, -1.35, -1.91, 
                  -0.30, -0.61, -1.08, -1.81, 
                  -0.11, -0.80, -1.19, -1.95, 
                  0, 0]
    """
    sim_ids = []
    potentials = []


    for i in range(len(sim_ids)):
        convert_xyz_to_pickle(xyz_file=f"data/xyz_files/Pt111_{sim_ids[i]}.xyz",
                          xyz_timestep=xyz_timestep,
                          cell_dims=cell_dims,
                          lattice_dims=lattice_dims,
                          electrode_potential=potentials[i]
                         )
 
    convert_xyz_to_pickle(xyz_file="data/xyz_files/example.xyz", 
                          xyz_timestep=xyz_timestep, 
                          cell_dims=cell_dims, 
                          lattice_dims=lattice_dims, 
                          electrode_potential=-0.12
                         )