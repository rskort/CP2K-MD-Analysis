import numpy as np

# -------------------------
# Class Definitions
# -------------------------

class Atom:
    """
    Atom class that stores the element and its x, y, z coordinates.
    """
    def __init__(self, element, x, y, z):
        self.element = element
        self.x = x
        self.y = y
        self.z = z

    def get_position(self):
        return [self.x, self.y, self.z]


class Frame:
    """
    Frame class that stores the simulation time (in fs),
    a list of Atom objects, water centers of mass, and the surface z.
    
    A Frame object now holds a reference to its parent Simulation, so that
    its metal type and lattice dimensions are always read from the parent.
    """
    def __init__(self, time, simulation=None):
        self.time = time            # Frame time in fs.
        self.atoms = []             # List of Atom objects.
        self.water_coms = []        # List to store water centers of mass.
        self.surface_z = None       # Will be computed later.
        self.simulation = simulation  # Reference to the parent Simulation.

    def add_atom(self, atom):
        # Adds an Atom object to the frame.
        self.atoms.append(atom)

    def compute_surface_z(self):
        """
        Compute the surface z value from metal atoms.
        The electrode is assumed to consist of metal atoms.
        The top layer is defined as the top 1/(number of layers) fraction of metal atoms
        when sorting by z in descending order.
        The surface_z is computed as the average z of these atoms.
        
        This method obtains the metal type and lattice dimensions from the parent Simulation.
        """
        # Check that a parent simulation is available.
        if self.simulation is None:
            raise ValueError("No parent Simulation assigned to Frame.")

        metal_type = self.simulation.metal_type
        lattice_dimensions = self.simulation.lattice_dimensions

        if metal_type is None:
            self.surface_z = 0.0
            return self.surface_z

        # Determine number of electrode layers from lattice_dimensions if provided.
        electrode_layers = lattice_dimensions[2] if lattice_dimensions is not None else 1

        # Filter metal atoms only.
        metal_atoms = [atom for atom in self.atoms if atom.element == metal_type]
        if not metal_atoms:
            self.surface_z = 0.0
            return self.surface_z

        # Sort metal atoms by z coordinate (highest first)
        metal_atoms.sort(key=lambda atom: atom.z, reverse=True)

        # Number of atoms in the top layer: total metal atoms divided by number of layers.
        n_top = len(metal_atoms) // electrode_layers
        if n_top < 1:
            n_top = len(metal_atoms)

        # Average the z coordinate of the top layer.
        top_layer_z = np.mean([atom.z for atom in metal_atoms[:n_top]])
        self.surface_z = round(top_layer_z, 4)
        return self.surface_z

    def compute_water_coms(self):
        """
        Compute the centers of mass (COM) of water molecules defined as one O atom 
        and its two closest H atoms within a given cutoff distance.
        
        The COM is computed as:
            COM = (m_O * r_O + m_H * r_H1 + m_H * r_H2) / (m_O + 2*m_H)
        where m_O=16 and m_H=1 (arbitrary units).

        Returns:
            self.water_coms (list): A list of numpy arrays, each the COM of a water molecule.
        """
        import numpy as np

        # Define atomic masses (arbitrary units)
        mass_O = 16.0
        mass_H = 1.0
        cutoff = 1.2  # distance cutoff in Angstrom (adjust as needed)

        water_coms = []

        # Build lists of indices for O and H atoms.
        O_indices = [i for i, atom in enumerate(self.atoms) if atom.element == 'O']
        H_indices = [i for i, atom in enumerate(self.atoms) if atom.element == 'H']

        # Get positions for all atoms once to speed up computations.
        positions = np.array([atom.get_position() for atom in self.atoms])

        # For each oxygen atom, search for hydrogen atoms within the cutoff.
        for o_idx in O_indices:
            O_pos = positions[o_idx]
            # Calculate distances from this oxygen to all hydrogen atoms.
            H_positions = positions[H_indices]
            distances = np.linalg.norm(H_positions - O_pos, axis=1)
            # Find indices (relative to H_indices) of hydrogen atoms within the cutoff.
            close_H = [ (H_indices[i], d) for i, d in enumerate(distances) if d <= cutoff ]
            if len(close_H) < 2:
                # Not enough H atoms found near this O; skip it.
                continue
            # Sort by distance and select the two closest hydrogen atoms.
            close_H.sort(key=lambda x: x[1])
            H1_idx, H2_idx = close_H[0][0], close_H[1][0]

            # Retrieve positions of the chosen H atoms.
            H1_pos = np.array(self.atoms[H1_idx].get_position())
            H2_pos = np.array(self.atoms[H2_idx].get_position())

            # Compute the center of mass.
            com = (mass_O * O_pos + mass_H * H1_pos + mass_H * H2_pos) / (mass_O + 2 * mass_H)
            water_coms.append(np.round(com, 4))

        self.water_coms = water_coms
        return self.water_coms



class Simulation:
    """
    Simulation class that stores a list of Frame objects, the project name,
    cell dimensions, MD timestep, and electrode (lattice) information.
    """
    def __init__(self, md_timestep, project_name, metal_type, 
                 cell_dimensions=None, lattice_dimensions=None):
        self.frames = []                                # List of Frame objects.
        self.project_name = project_name.replace('_', ' ')
        self.cell_dimensions = cell_dimensions          # Tuple of (Lx, Ly, Lz) if provided.
        self.md_timestep = md_timestep                  # MD timestep in fs.
        self.metal_type = metal_type                    # e.g. "Pt" or "Au"
        self.lattice_dimensions = lattice_dimensions    # Tuple of (latX, latY, layers) if provided.

    def add_frame(self, frame):
        # Assign the parent simulation to the frame.
        frame.simulation = self
        # Now compute the surface_z and water centers of mass.
        frame.compute_surface_z()
        frame.compute_water_coms()
        self.frames.append(frame)
