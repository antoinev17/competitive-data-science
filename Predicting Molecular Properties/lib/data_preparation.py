import numpy as np

def create_coupling_batch(molecule_structures, coupled_atoms, COUPLING_TYPE_NB = 8, MAX_MOL_ATOMS_NB = 29):
    
    ### coupled_atomes has 4 columns: molecule_name, atom_index_0, atom_index_1, type

    batch_size = coupled_atoms.shape[0]
    
    atom_col = ['x', 'y', 'z', 'atom_C','atom_F','atom_H','atom_N','atom_O']
    drop_col = ['molecule_name', 'atom_index'] 
    
    # Select all molecules in the batch first (to quicken the operations)
    molecules_name = coupled_atoms[['molecule_name']].values.reshape(-1)
    molecules = molecule_structures.loc[molecule_structures['molecule_name'].isin(molecules_name)]
    
    # create the output matrices  
    coupled_atoms_feat = np.zeros((batch_size, 8 * 2 + COUPLING_TYPE_NB))
    atom_pos_feat = np.zeros((batch_size, MAX_MOL_ATOMS_NB, len(atom_col)))
    
    # Loop over all samples in the batch
    for i in np.arange(batch_size):
        molecule_name = coupled_atoms.iloc[i,0]
        atom_index_A  = coupled_atoms.iloc[i,1]
        atom_index_B  = coupled_atoms.iloc[i,2]
        coupling_type = coupled_atoms.iloc[i,-COUPLING_TYPE_NB:].values.reshape(1,COUPLING_TYPE_NB )
        
        molecule = molecules.loc[(molecules['molecule_name'] == molecule_name)]
        atom_1 = molecule.loc[(molecule['atom_index'] == atom_index_A)].drop(drop_col, axis = 1).values
        atom_2 = molecule.loc[(molecule['atom_index'] == atom_index_B)].drop(drop_col, axis = 1).values

        coupled_atoms_feat[i,:] = np.concatenate((atom_1, atom_2, coupling_type ), axis = 1)
        
        # calculate the mean position between the 2 coupled_atoms
        p_mean = molecule.loc[(molecule['atom_index'] == atom_index_A) |
                             (molecule['atom_index'] == atom_index_B), ['x', 'y', 'z']].mean(axis = 0)
        
        # retrieve the atom features of all atoms in the molecule
        mol_atom_pos = molecule[atom_col].copy()
           
        mol_atom_pos['distance'] = mol_atom_pos.apply(lambda row : 
                                  np.sqrt((row['x']-p_mean['x'])**2 + (row['y']-p_mean['y'])**2 + (row['z']-p_mean['z'])**2)
                                  , axis = 1)
        mol_atom_pos = mol_atom_pos.sort_values('distance',ascending = False).drop(['distance'], axis = 1)
        atom_pos_feat[i, :mol_atom_pos.shape[0], :mol_atom_pos.shape[1]] = mol_atom_pos.values
    
    return coupled_atoms_feat, atom_pos_feat

print('hello world')