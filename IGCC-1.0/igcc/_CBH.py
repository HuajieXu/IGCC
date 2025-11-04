# -*- coding: utf-8 -*-
from math import ceil
from re import findall
from rdkit import Chem
from copy import deepcopy
from os.path import exists
from functools import lru_cache
from collections import Counter
from pymatgen.core import Molecule
from itertools import combinations, chain, product
from rdkit.Chem import AllChem, Descriptors, rdchem

from igcc._basic import Basic
from igcc._conformer import Conformer


class CBH(object):
    def __init__(self, para):
        self.para = para
        self.basic = Basic(para)
    
    @lru_cache(maxsize=128)
    def encode_to_number(self, num_tuple):
        encoded_str = ''

        for x in num_tuple:
            encoded_str += str(int(x) + 1)
        encoded_num = int(encoded_str)

        return encoded_num
    
    @lru_cache(maxsize=128)
    def decode_to_list(self, encoded_num):
        decoded_list = []
    
        for x in str(encoded_num):
            decoded_list.append(int(x) - 1)
    
        return decoded_list
   
    @lru_cache(maxsize=128)
    def get_ori_mol(self, smi):
        cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
        
        # Process the structure of electronic-containing molecules separately.
        ori_mol = Chem.MolFromSmarts(cal_smi)
        if [v for k, v in self.para['charge_group'].items() if list(ori_mol.GetSubstructMatches(Chem.MolFromSmarts(k)))]:
            mol = Chem.AddHs(Chem.MolFromSmiles(cal_smi))
            molblock = Chem.MolToMolBlock(mol)

        # Read the ready existed structure for completed molecules.
        elif exists(f"{self.para['work_path']}/rawfiles/B3LYP/{inc}.out") and self.para['geometry_state'].get(inc, ['', ''])[:-2] == std_smi:
            mol = Molecule.from_file(f"{self.para['work_path']}/rawfiles/B3LYP/{inc}.out")
            molblock = mol.to(fmt='mol')


        # Build conformational structure for unknown molecules.
        else:
            gjf = Conformer(inc, self.para, 2).output_standard_gjf(std_smi)
            mol = Molecule.from_str(gjf, fmt='gjf')
            molblock = mol.to(fmt='mol')
            cur_smi = Chem.MolToSmiles(Chem.MolFromSmiles(mol.to(fmt='smiles').strip()))

            # Make the current molecular structure identical to that of the target smiles.
            if cur_smi != cal_smi:
                mol = Chem.AddHs(Chem.MolFromSmiles(cal_smi))
                molblock = Chem.MolToMolBlock(mol)

        mol = Chem.MolFromMolBlock(molblock, removeHs=False)

        return mol
    
    @lru_cache(maxsize=128)
    def get_end_count(self, data):
        data = dict(data)
        atoms = Counter()
        
        for k, v in data.items():
            cal_smi_k, std_smi_k, _ = self.basic.smi_to_std_format(k)
            mol_atoms = Counter([x.GetSymbol() for x in Chem.AddHs(Chem.MolFromSmiles(cal_smi_k)).GetAtoms()])
            atoms.update({x: y * v for x, y in mol_atoms.items()})
        
        return atoms

    
    """ To get CBH count. """
    @lru_cache(maxsize=128)
    def CBH_count(self, smi, rung):
        
        cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
        bonds, smiles = Counter(), Counter()
        
        if int(rung) == 0:
            mol = self.basic.get_mol_structure(std_smi, removeHs=True)
            for atom in mol.GetAtoms():
                retain_atoms = (atom.GetIdx(),)
                bond_smi, mole_smi = self.get_fragment_smiles(smi, mol, retain_atoms, 'low')
                bonds.update(bond_smi), smiles.update(mole_smi)
        else:
            mol = self.basic.get_mol_structure(std_smi, removeHs=True)
            smallest_rings = self.get_smallest_rings(mol)

            if smallest_rings:
                bond_smi, mole_smi = self.separate_multiple_rings(std_smi, mol, smallest_rings)
                
                if int(rung) == 1:
                    bonds = bonds + bond_smi
                
                for k, v in mole_smi.items():
                    cal_smi_k, std_smi_k, _ = self.basic.smi_to_std_format(k)
                    mol_k = self.basic.get_mol_structure(std_smi_k, removeHs=True)
                    smallest_rings_k = self.get_smallest_rings(mol_k)

                    
                    if smallest_rings_k:
                        if int(rung) == 1:
                            if len(list(mol_k.GetAromaticAtoms())) > 0:
                                retain_atoms = tuple(range(mol_k.GetNumAtoms()))
                                bond_smi_k, mole_smi_k = self.get_fragment_smiles(std_smi_k, mol_k, retain_atoms, 'low')
                                bonds = bonds + Counter({m: v * n for m, n in Counter(bond_smi_k).items()})
                                smiles = smiles + Counter({m: v * n for m, n in Counter(mole_smi_k).items()})
                            else:
                                for bond in mol_k.GetBonds():
                                    retain_atoms = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                                    bond_smi_k, mole_smi_k = self.get_fragment_smiles(std_smi_k, mol_k, retain_atoms, 'low')
                                    bonds = bonds + Counter({m: v * n for m, n in Counter(bond_smi_k).items()})
                                    smiles = smiles + Counter({m: v * n for m, n in Counter(mole_smi_k).items()})

                        elif int(rung) == 2:
                            if 0 < len(list(mol_k.GetAromaticAtoms())) != 3:
                                retain_atoms = tuple(range(mol_k.GetNumAtoms()))
                                _, mole_smi_k = self.get_fragment_smiles(std_smi_k, mol_k, retain_atoms, 'low')
                                smiles = smiles + Counter({m: v * n for m, n in Counter(mole_smi_k).items()})
                            elif len(smallest_rings_k[0]) == 3:
                                retain_atoms = tuple(range(mol_k.GetNumAtoms()))
                                bond_smi_k, mole_smi_k = self.get_fragment_smiles(std_smi_k, mol_k, retain_atoms, 'low')
                                bonds = bonds + Counter({m: v * n for m, n in Counter(bond_smi_k).items()})
                                smiles = smiles + Counter({m: v * n for m, n in Counter(mole_smi_k).items()})
                            else:
                                bond_smi_k, mole_smi_k = self.get_old_CBH_count(std_smi_k, mol_k, rung)
                                bonds = bonds + Counter({m: v * n for m, n in Counter(bond_smi_k).items()})
                                smiles = smiles + Counter({m: v * n for m, n in Counter(mole_smi_k).items()})

                        elif int(rung) == 3:
                            if 0 < len(list(mol_k.GetAromaticAtoms())) != 4 or len(smallest_rings_k[0]) == 3:
                                retain_atoms = tuple(range(mol_k.GetNumAtoms()))
                                _, mole_smi_k = self.get_fragment_smiles(std_smi_k, mol_k, retain_atoms, 'low')
                                smiles = smiles + Counter({m: v * n for m, n in Counter(mole_smi_k).items()})
                            elif len(smallest_rings_k[0]) == 4:
                                retain_atoms = tuple(range(mol_k.GetNumAtoms()))
                                bond_smi_k, mole_smi_k = self.get_fragment_smiles(std_smi_k, mol_k, retain_atoms, 'low')
                                bonds = bonds + Counter({m: v * n for m, n in Counter(bond_smi_k).items()})
                                smiles = smiles + Counter({m: v * n for m, n in Counter(mole_smi_k).items()})
                            else:
                                bond_smi_k, mole_smi_k = self.get_old_CBH_count(std_smi_k, mol_k, rung)
                                bonds = bonds + Counter({m: v * n for m, n in Counter(bond_smi_k).items()})
                                smiles = smiles + Counter({m: v * n for m, n in Counter(mole_smi_k).items()})
                    else:
                        bond_smi_k, mole_smi_k = self.get_old_CBH_count(std_smi_k, mol_k, rung)
                        
                        if int(rung) == 1:
                            bonds = bonds + Counter({m: v * n for m, n in Counter(bond_smi_k).items()})
                            smiles = smiles + Counter({m: v * n for m, n in Counter(mole_smi_k).items()})
                        else:
                            smiles = smiles + Counter({m: v * n for m, n in Counter(mole_smi_k).items()})

            else:
                bond_smi, mole_smi = self.get_old_CBH_count(std_smi, mol, rung)
                bonds = bonds + bond_smi
                smiles = smiles + mole_smi
        
        return bonds, smiles
    
    @lru_cache(maxsize=128)
    def get_old_CBH_count(self, smi, mol, rung):
        bonds, smiles = [], []
        mol = self.charge_to_single_atom(mol)

        if int(rung) == 1:
            retain_atoms = tuple(range(mol.GetNumAtoms()))
            bond_smi, mole_smi = self.get_fragment_smiles(smi, mol, retain_atoms, 'low')
            
            if mol.GetNumHeavyAtoms() <= 2:
                retain_atoms = tuple(range(mol.GetNumAtoms()))
                bond_smi, mole_smi = self.get_fragment_smiles(smi, mol, retain_atoms, 'low')
                bonds.extend(bond_smi), smiles.extend(mole_smi)
  
            else:
                for bond in mol.GetBonds():
                    retain_atoms = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    bond_smi, mole_smi = self.get_fragment_smiles(smi, mol, retain_atoms, 'low')
                    bonds.extend(bond_smi), smiles.extend(mole_smi)
        
        elif int(rung) == 2:
            if mol.GetNumHeavyAtoms() <= 3:
                retain_atoms = tuple(range(mol.GetNumAtoms()))
                bond_smi, mole_smi = self.get_fragment_smiles(smi, mol, retain_atoms, 'low')
                
                if mol.GetNumHeavyAtoms() < 3:
                    smiles.extend(mole_smi)
                else:
                    bonds.extend(bond_smi), smiles.extend(mole_smi)

            else:
                for atom in mol.GetAtoms():
                    neighbor_atoms = [x.GetIdx() for x in atom.GetNeighbors()]
                        
                    if len(neighbor_atoms) > 1:
                        retain_atoms = tuple(set(neighbor_atoms + [atom.GetIdx()]))
                        bond_smi, mole_smi = self.get_fragment_smiles(smi, mol, retain_atoms, 'low')
                        bonds.extend(bond_smi), smiles.extend(mole_smi)
        
        elif int(rung) == 3:
            if mol.GetNumHeavyAtoms() <= 4:
                retain_atoms = tuple(range(mol.GetNumAtoms()))
                bond_smi, mole_smi = self.get_fragment_smiles(smi, mol, retain_atoms, 'low')
                
                if mol.GetNumHeavyAtoms() < 4:
                    smiles.extend(mole_smi)
                else:
                    bonds.extend(bond_smi), smiles.extend(mole_smi)

            else:
                for bond in mol.GetBonds():
                    left_atoms =  [x.GetIdx() for x in bond.GetBeginAtom().GetNeighbors() if x.GetIdx() != bond.GetEndAtomIdx()]
                    right_atoms =  [x.GetIdx() for x in bond.GetEndAtom().GetNeighbors() if x.GetIdx() != bond.GetBeginAtomIdx()]

                    if left_atoms and right_atoms and set(left_atoms) != set(right_atoms):
                        retain_atoms = tuple(set(left_atoms + right_atoms + [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
                        bond_smi, mole_smi = self.get_fragment_smiles(smi, mol, retain_atoms, 'low')
                        bonds.extend(bond_smi), smiles.extend(mole_smi)

        if not bonds and not smiles:
            retain_atoms = tuple(range(mol.GetNumAtoms()))
            bond_smi, mole_smi = self.get_fragment_smiles(smi, mol, retain_atoms, 'low')
            bonds.extend(bond_smi), smiles.extend(mole_smi)

        bonds, smiles = Counter(bonds), Counter(smiles)

        return bonds, smiles


    @lru_cache(maxsize=128)
    def get_rung_count(self, data, rung):
        data = dict(data)
        all_smiles, rung = Counter(), int(rung)
        
        for k, v in data.items():
            _, smiles = self.CBH_count(k, rung)
            all_smiles = all_smiles + Counter({m: v * n for m, n in smiles.items()})

        return all_smiles

    """ To get all count bonds of each side for a given rung. """
    @lru_cache(maxsize=128)
    def get_bonds_count(self, smi, rung):
        all_bonds, rung = Counter({}), int(rung)

        for i in range(1, rung + 1):
            bonds, _ = eval(f"self.CBH_count('{smi}', {i})")
            all_bonds = all_bonds + bonds

        return all_bonds
    
    @lru_cache(maxsize=128)
    def charge_to_single_atom(self, mol):
        result_mol = Chem.RWMol(mol)
        for k, v in self.para['charge_group'].items():
            while True:
                charge_pair = [set(x) for x in result_mol.GetSubstructMatches(Chem.MolFromSmarts(k))]
                if not charge_pair: break
                charge_atoms = charge_pair[0]
                retain_atoms = set(x.GetIdx() for x in result_mol.GetAtoms()) - charge_atoms
                all_ends = [[x, y.GetIdx()] for x in retain_atoms for y in result_mol.GetAtomWithIdx(x).GetNeighbors() if y.GetIdx() in charge_atoms]

                if len(all_ends) == 0:
                    result_mol.AddAtom(Chem.Atom(0))
                    result_mol.GetAtomWithIdx(result_mol.GetNumAtoms() - 1).SetIsotope(v)
                elif len(all_ends) == 1:
                    both_ends = all_ends[0]
                    result_mol.AddAtom(Chem.Atom(0))
                    result_mol = self.add_virtual_bond(result_mol, result_mol, tuple(both_ends), v)
                else:
                    result_mol.AddAtom(Chem.Atom(0))
                    for both_ends in all_ends:
                        result_mol = self.add_virtual_bond(result_mol, result_mol, tuple(both_ends), v)

                for atom in sorted(charge_atoms, reverse = True):
                    result_mol.RemoveAtom(atom)
        
        return result_mol



    """ To separate multiple rings for policyclic species. """
    @lru_cache(maxsize=128)
    def separate_multiple_rings(self, smi, mol, smallest_rings):
        bonds, smiles = [], []
        smallest_rings = list(map(set, smallest_rings))
        ring_atoms = tuple(set(list(chain(*smallest_rings))))
        non_ring_atoms = tuple(set(range(mol.GetNumAtoms())) - set(ring_atoms))
        bridged_bonds = [sorted(x & y) for x, y in combinations(smallest_rings, 2) if len(x & y) == 2]

        for ring in smallest_rings:
            retain_atoms = tuple(ring)
            bond_smi, mole_smi = self.get_fragment_smiles(smi, mol, retain_atoms, 'low', self.basic.list_set_to_tuple(bridged_bonds))
            bonds.extend(bond_smi)
            smiles.extend(mole_smi)
        
        retain_atoms = non_ring_atoms
        bond_smi, _ = self.get_fragment_smiles(smi, mol, retain_atoms, 'low', self.basic.list_set_to_tuple(bridged_bonds))

        for x in bond_smi:
            if x:
                smiles.extend(self.bond_smi_to_mole_smi(x, 'high').split('.'))

        return Counter(bonds), Counter(smiles)
    
    @lru_cache(maxsize=128)
    def get_smallest_rings(self, mol):
        smallest_rings = []
        rings = [set(x) for x in Chem.GetSymmSSSR(mol)]

        aromatic_ring = sorted(self.para['aromatic_group'], key=lambda x: self.basic.get_all_atoms(x), reverse=True)

        for ring in aromatic_ring:
            for x in mol.GetSubstructMatches(Chem.MolFromSmarts(ring)):
                if all([not set(x).issubset(y) for y in smallest_rings]):
                    smallest_rings.append(set(x))

        all_atoms = set(chain(*smallest_rings))
        for x in rings:
            if x - all_atoms:
                smallest_rings.append(x)
        
        smallest_rings = self.basic.list_set_to_tuple(smallest_rings)
        
        return smallest_rings


    """ To get CBH reactions. """
    @lru_cache(maxsize=128)
    def get_CBH_reactions(self, smi, rung):
        cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)

        left, right = self.supply_multiple_rings(smi)

        reac =  Counter([std_smi]) + left
        prod =  self.CBH_count(std_smi, rung)[-1] + right

        reac, prod, _ = self.supply_reduced_rungs(self.basic.dict_to_tuple(reac), self.basic.dict_to_tuple(prod), rung)

        res_reac = reac - prod
        res_prod = prod - reac

        left = self.get_end_count(self.basic.dict_to_tuple(res_reac))
        right = self.get_end_count(self.basic.dict_to_tuple(res_prod))

        if left != right:
            print(f'!!! ERROR: The isodesmic reaction is not balanced: --> {inc}: {std_smi}\nThe program exit abnormally.')
            exit()

        elif not left and not right:
            res_reac = Counter([std_smi])
            res_prod = Counter([std_smi])
        
        reac_form = f"{' + '.join([f'{res_reac[x]} {x}' for x in sorted(res_reac, key=lambda x: self.basic.get_all_atoms(x), reverse=True)])}"
        prod_form = f"{' + '.join([f'{res_prod[x]} {x}' for x in sorted(res_prod, key=lambda x: self.basic.get_all_atoms(x), reverse=True)])}"
        reaction =  f'{reac_form} --> {prod_form}'
        
        return res_reac, res_prod, reaction


    """ To get CBH delta bonds. """
    @lru_cache(maxsize=128)
    def get_CBH_delta_bonds(self, smi, rung):
        cal_smi, std_smi, _ = self.basic.smi_to_std_format(smi)

        reac, prod, _ = self.get_CBH_reactions(std_smi, rung)
        reac_bonds, prod_bonds = Counter({}), Counter({})
        
        for k, v in reac.items():
            bonds = self.get_bonds_count(k, rung)
            reac_bonds = reac_bonds + Counter({m: v * n for m, n in bonds.items()})

        for k, v in prod.items():
            bonds = self.get_bonds_count(k, rung)
            prod_bonds = prod_bonds + Counter({m: v * n for m, n in bonds.items()})
        
        delta_bonds = Counter({x: prod_bonds.get(x, 0) - reac_bonds.get(x, 0) for x in set(reac_bonds) | set(prod_bonds) if reac_bonds.get(x, 0) - prod_bonds.get(x, 0)})

        return reac_bonds, prod_bonds, delta_bonds


    """ To supply multiple aromatic rings from the left side. """
    @lru_cache(maxsize=128)
    def supply_multiple_rings(self, smi):
        cal_smi, std_smi, _ = self.basic.smi_to_std_format(smi)
        reac, prod, mol = [], [], Chem.MolFromSmiles(cal_smi)
        smallest_rings =  list(map(set, self.get_smallest_rings(mol)))
        bridged_bonds = [sorted(x & y) for x, y in combinations(smallest_rings, 2) if len(x & y) == 2]
        mutual_atoms = [x for x in Counter(chain(*bridged_bonds)) if Counter(chain(*bridged_bonds)).get(x, 0) >= 3]
        
        for bond_pair in bridged_bonds:
            result_mol = Chem.RWMol(mol)
            bond =  result_mol.GetBondBetweenAtoms(*bond_pair)
            bridged_rings = [x for x in smallest_rings if set(bond_pair).issubset(x)]
            aromatic_check = [all([result_mol.GetAtomWithIdx(y).GetIsAromatic() for y in x]) for x in bridged_rings]
            
            bond.GetBeginAtom().SetIsAromatic(False)
            bond.GetEndAtom().SetIsAromatic(False)
            
            if bond.GetBondTypeAsDouble() == 1.5 and all(aromatic_check):
                bond.SetBondType(Chem.BondType.DOUBLE)
            elif bond.GetBondTypeAsDouble() == 1.5 and not all(aromatic_check):
                bond.SetBondType(Chem.BondType.SINGLE)

            remove_atoms = set(x.GetIdx() for x in result_mol.GetAtoms()) - set(bond_pair)
            for atom in sorted(remove_atoms, reverse = True):
                result_mol.RemoveAtom(atom)

            mole_smi = Chem.MolToSmiles(result_mol)
            reac.append(mole_smi)

        for atom in mutual_atoms:
            result_mol = Chem.RWMol(mol)
            result_mol.GetAtomWithIdx(atom).SetIsAromatic(False)
            mole_smi = result_mol.GetAtomWithIdx(atom).GetSmarts()
            prod.append(mole_smi)

        return Counter(reac), Counter(prod)


    """ To supply molecules for reaction from reduced rungs. """
    @lru_cache(maxsize=128)
    def supply_reduced_rungs(self, reac, prod, rung):
        reac, prod = Counter(dict(reac)), Counter(dict(prod))

        left = self.get_end_count(self.basic.dict_to_tuple(reac))
        right = self.get_end_count(self.basic.dict_to_tuple(prod))

        if {x: 0 for x in list(left) + list(right) if left.get(x, 0) - right.get(x, 0)} == {'H': 0} or rung == 0:
            if left < right:
                supply = self.supply_simple_molecules(self.basic.dict_to_tuple(right - left))
                reac = reac + supply
            elif left > right:
                supply = self.supply_simple_molecules(self.basic.dict_to_tuple(left - right))
                prod = prod + supply

        elif left < right and rung:
            CBH_left = self.get_rung_count(self.basic.dict_to_tuple(reac), rung - 1)
            CBH_right = self.get_rung_count(self.basic.dict_to_tuple(prod), rung - 1)
            reac = reac + Counter({k: v for k, v in (CBH_right - CBH_left).items() if v != 0 and k in CBH_left})
            reac, prod, rung = self.supply_reduced_rungs(self.basic.dict_to_tuple(reac), self.basic.dict_to_tuple(prod), rung - 1)

        
        elif left > right and rung:
            CBH_left = self.get_rung_count(self.basic.dict_to_tuple(reac), rung - 1)
            CBH_right = self.get_rung_count(self.basic.dict_to_tuple(prod), rung - 1)
            prod = prod + Counter({k: v for k, v in (CBH_left - CBH_right).items() if v != 0 and k in CBH_right})
            reac, prod, rung = self.supply_reduced_rungs(self.basic.dict_to_tuple(reac), self.basic.dict_to_tuple(prod), rung - 1)

        return reac, prod, rung


    """ To supply simple molecules. """
    @lru_cache(maxsize=128)
    def supply_simple_molecules(self, residual):
        residual = dict(residual)
        
        if set(residual) == set('CHO'):
            molecules = Counter({'C': residual['C']}) + Counter({'O': residual['O']}) + Counter({'[H][H]': int(0.5 * residual['H'] + 2 * residual['C'] - residual['O'])})
        elif set(residual) == set('CH'):
            molecules = Counter({'C': residual['C']}) + Counter({'[H][H]': int(0.5 * residual['H'] - 2 * residual['C'])})
        elif set(residual) == set('HO'):
            molecules = Counter({'O': residual['O']}) + Counter({'[H][H]': int(0.5 * residual['H'] - residual['O'])})
        elif set(residual) ==  set('H'):
            molecules= Counter({'[H][H]': int(0.5 * residual['H'])})
        else:
            molecules = Counter([])
        
        return molecules
    
    #@lru_cache(maxsize=128)
    def add_methyl_to_mol(self, atom_idx, mol):
        mol.AddAtom(Chem.Atom(6))
        other_atom_idx = mol.GetNumAtoms() - 1
        mol.AddBond(atom_idx, mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)
        
        for i in range(3):
            mol.AddAtom(Chem.Atom(1))
            mol.AddBond(other_atom_idx, mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)

        return mol


    @lru_cache(maxsize=128)
    def bond_smi_to_mole_smi(self, bond_smi, flag):
        try:
            if len(bond_smi) > 2 and findall('^[CTZE]-', bond_smi):
                cal_bond_smi = Chem.MolToSmiles(Chem.MolFromSmiles(bond_smi[2:], sanitize=False))
            else:
                cal_bond_smi = Chem.MolToSmiles(Chem.MolFromSmiles(bond_smi, sanitize=False))
        except:
            return None
        
        # Replace the special group.
        mol = Chem.MolFromSmiles(cal_bond_smi, sanitize=False)
        
        atom_radical_electrons = {}
        result_mol = Chem.RWMol(mol)
        remove_atoms = []

        for atom in mol.GetAtoms():
            explict_Hs = atom.GetNumExplicitHs()

            if atom.GetAtomicNum() == 0:
                if flag == 'high':
                    atom_isotope = atom.GetIsotope()

                    result_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(atom_isotope)
                    result_mol.GetAtomWithIdx(atom.GetIdx()).SetIsotope(0)

                    try:
                        total_valence = atom.GetTotalValence()
                    except:
                        total_valence = round(sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()])) + explict_Hs

                    if 1 < atom_isotope < 100:
                        current_non_H_valence = total_valence - explict_Hs
                        atom_hybridization, non_H_valence, radical_electrons, atom_electrons = self.decode_to_list(atom.GetAtomMapNum())
                        total_Hs = non_H_valence - current_non_H_valence + explict_Hs
                        atom_radical_electrons.update({atom.GetIdx(): atom_electrons})
                        result_mol.GetAtomWithIdx(atom.GetIdx()).SetNumExplicitHs(explict_Hs)
                        result_mol.GetAtomWithIdx(atom.GetIdx()).SetNumRadicalElectrons(radical_electrons)

                        if any([bond.GetIsAromatic() for bond in atom.GetBonds()]) and atom.IsInRing():
                            result_mol.GetAtomWithIdx(atom.GetIdx()).SetIsAromatic(True)
                            
                            for bond in atom.GetBonds():
                                result_mol.GetBondWithIdx(bond.GetIdx()).SetIsAromatic(True)
                        else:
                            extra_non_H_valence = non_H_valence - current_non_H_valence
                            if extra_non_H_valence:
                                for i in range(extra_non_H_valence):
                                    result_mol = self.add_methyl_to_mol(atom.GetIdx(), result_mol)  
                else:
                    remove_atoms.append(atom.GetIdx())
            else:
                if flag == 'low':
                    if atom.GetAtomicNum() > 1:
                        aromatic_ring = set(chain(*[x for x in self.get_smallest_rings(mol) if all([mol.GetAtomWithIdx(y).GetIsAromatic() for y in x]) and all([mol.GetAtomWithIdx(y).GetAtomicNum() > 1 for y in x])]))

                        if atom.GetIdx() in aromatic_ring:
                            try:
                                total_Hs = atom.GetTotalNumHs()
                            except:
                                total_Hs = [x.GetSymbol() for x in atom.GetNeighbors()].count('H')
                        else:
                            result_mol.GetAtomWithIdx(atom.GetIdx()).SetIsAromatic(False)
                        
                            for bond in atom.GetBonds():
                                if bond.GetOtherAtomIdx(atom.GetIdx()) not in aromatic_ring and bond.GetBondTypeAsDouble() == 1.5:
                                    result_mol.GetBondWithIdx(bond.GetIdx()).SetBondType(Chem.BondType.SINGLE)
                        
                            atom_electrons = atom.GetNumRadicalElectrons()
                            non_H_valence = int(sum([x.GetBondTypeAsDouble() for x in atom.GetBonds() if x.GetOtherAtom(atom).GetAtomicNum() > 1])) - atom.GetFormalCharge()
                            total_Hs = {1: 1, 6: 4, 8: 2}.get(atom.GetAtomicNum()) - atom_electrons - non_H_valence
                        result_mol.GetAtomWithIdx(atom.GetIdx()).SetNumExplicitHs(total_Hs)

            if atom.GetAtomMapNum():
                atom_hybridization, non_H_valence, radical_electrons, atom_electrons = self.decode_to_list(mol.GetAtomWithIdx(atom.GetIdx()).GetAtomMapNum())
                atom_radical_electrons.update({atom.GetIdx(): atom_electrons})
                result_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0)
                result_mol.GetAtomWithIdx(atom.GetIdx()).SetNumExplicitHs(explict_Hs)
                result_mol.GetAtomWithIdx(atom.GetIdx()).SetNumRadicalElectrons(radical_electrons)

        for atom in sorted(remove_atoms, reverse=True):
            result_mol.RemoveAtom(atom)
        
        try:
            mole_smi = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(result_mol)))
        except:
            return None
        
        if not any([x in mole_smi for x in self.para['charge_group']]) and mole_smi != '[C-]#[O+]':
            mole_smi = mole_smi.replace('+]', ']').replace('-]', ']')

        if flag == 'high':
            spin = abs(sum(atom_radical_electrons.values())) + 1
        else:
            spin = abs(sum(atom_radical_electrons.values()) - sum([atom_radical_electrons.get(x, 0) for x in remove_atoms])) + 1

        mol_spin = Chem.Descriptors.NumRadicalElectrons(Chem.MolFromSmiles(mole_smi)) + 1
        
        if spin < mol_spin and abs(spin - mol_spin) % 2 == 0:
            mole_smi = f'{mole_smi}-{spin}'
        elif spin > mol_spin and abs(spin - mol_spin) % 2 == 0:
            mole_smi = f'{mole_smi}-{spin}'

        if len(bond_smi) > 2 and findall('^[CTZE]-', bond_smi):
            mole_smi = f'{bond_smi[:2]}{mole_smi}'

        return mole_smi


    @lru_cache(maxsize=128)
    def get_bridged_rings(self, mol, ring_atoms, aromatic_rings):
        ring_atoms, aromatic_rings = set(ring_atoms), set(aromatic_rings)
        bridged_rings = set()
        
        for atom_idx in ring_atoms:
            for other_atom in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                for ring in aromatic_rings:
                    if other_atom.GetIdx() in ring and not mol.GetBondBetweenAtoms(atom_idx, other_atom.GetIdx()).GetIsAromatic():
                        bridged_rings.add(ring)

        return bridged_rings
    
    @lru_cache(maxsize=128)
    def get_adjcent_rings(self, ring_atoms, aromatic_rings):
        ring_atoms, aromatic_rings = set(ring_atoms), set(aromatic_rings)
        adjcent_rings = set([x for x in aromatic_rings if len(set(x) & ring_atoms) == 2])

        return adjcent_rings



    @lru_cache(maxsize=128)
    def test_aromatic_mol(self, mol, all_atoms):
        aromatic_atoms = set(all_atoms)
        result_mol = Chem.RWMol(mol)
        remove_atoms = sorted(set(range(result_mol.GetNumAtoms())) - set(all_atoms), reverse=True)
        
        for x in remove_atoms:
            result_mol.RemoveAtom(x)

        smi = Chem.MolToSmiles(result_mol)
        try:
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            aromatic_check = True
        except:
            aromatic_check = False

        return aromatic_check


    """ To get smiles of fragments. """
    @lru_cache(maxsize=128)
    def get_fragment_smiles(self, smi, mol, retain_atoms, flag='low', bridged_bonds=tuple()):
        max_valence = {'C': 4, 'O': 2}
        retain_atoms, bridged_bonds = set(retain_atoms), bridged_bonds
        bonds, smiles, mol, result_mol = [], [], Chem.RWMol(mol), Chem.RWMol(mol)

        atom_radical_electrons = self.basic.get_atom_radical_electrons(smi, AddHs=False)
        extra_atoms = set(y.GetIdx() for x in retain_atoms for y in mol.GetAtomWithIdx(x).GetNeighbors() if y.GetIdx() not in retain_atoms)
        smallest_rings = self.get_smallest_rings(mol)
        aromatic_atoms = set([x.GetIdx() for x  in mol.GetAromaticAtoms()])
        aromatic_rings = set([tuple(sorted(x)) for x in smallest_rings if set(x).issubset(aromatic_atoms)])
        original_rings = set([x for x in smallest_rings if set(x).issubset(set([y for y in retain_atoms if mol.GetAtomWithIdx(y).IsInRing()]))])
        all_mutual_atoms = set([k for k, v in Counter(chain(*smallest_rings)).items() if v >=3])
        
        if original_rings:
            bridged_rings = self.get_bridged_rings(mol, tuple(chain(*original_rings)), tuple(aromatic_rings))
            adjcent_rings = self.get_adjcent_rings(tuple(chain(*original_rings)), tuple(aromatic_rings))
            all_rings = original_rings | bridged_rings | adjcent_rings
            all_atoms = list(chain(*[x for x in all_rings]))
            count = 0

            while not self.test_aromatic_mol(mol, tuple(all_atoms)):
                mutual_atoms = [k for k, v in Counter(all_atoms).items() if v >= 3]
                if count == 3:
                    mutual_atoms = list(set(mutual_atoms + [k for k, v in Counter(chain(*aromatic_rings)).items() if v >= 3]))
                end_atoms = set([y.GetIdx() for x in mutual_atoms for y in mol.GetAtomWithIdx(x).GetNeighbors()])
                end_rings = set([x for x in smallest_rings if set(x) & set(all_atoms) & all_mutual_atoms])
                all_rings = all_rings | end_rings
                all_atoms = list(chain(*[x for x in all_rings]))
                count += 1
                if count == 5:
                    print(f'!!! ERROR: Imbalanced aromatic smiles: --> {smi}\nThe program exit abnormally.')
                    exit()
            
            extra_atoms = extra_atoms | set(all_atoms) - retain_atoms

        end_atoms = set()

        for atom_idx in extra_atoms:
            for other_atom in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                if mol.GetBondBetweenAtoms(atom_idx, other_atom.GetIdx()).GetBondTypeAsDouble() > 1.5:
                    end_atoms.add(other_atom.GetIdx())
                elif mol.GetBondBetweenAtoms(atom_idx, other_atom.GetIdx()).GetBondTypeAsDouble() == 1.5:
                    retain_bond = [(x, atom_idx) for x in retain_atoms if mol.GetBondBetweenAtoms(x, atom_idx) and mol.GetBondBetweenAtoms(x, atom_idx).GetBondTypeAsDouble() == 1]
                    if retain_bond:
                        end_atoms.update(set(chain(*[x for x in aromatic_rings if set([atom_idx, other_atom.GetIdx()]).issubset(set(x))])))

        extra_atoms = extra_atoms | end_atoms - retain_atoms
 
        remove_atoms = sorted({x.GetIdx() for x in mol.GetAtoms()} - retain_atoms - extra_atoms, reverse = True)

        for atom_idx in extra_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_isotope = atom.GetAtomicNum()
            total_valence = atom.GetTotalValence()
            explict_Hs = atom.GetTotalNumHs() + sum(int(x.GetBondTypeAsDouble()) for x in atom.GetBonds() if x.GetOtherAtomIdx(atom_idx) not in retain_atoms | extra_atoms)
            
            if 1 < atom_isotope < 100:
                non_H_valence = total_valence - explict_Hs
                atom_hybridization =  self.basic.get_atom_hybration(atom, self.basic.dict_to_tuple(atom_radical_electrons))
                radical_electrons = atom.GetNumRadicalElectrons()
                atom_electrons = atom_radical_electrons.get(atom_idx, 0)
                
                # Get accurate hybridization for outside atom.
                atom_hybridization += max([int(x.GetBondTypeAsDouble()) for x in atom.GetBonds() if x.GetOtherAtomIdx(atom_idx) not in retain_atoms | extra_atoms], default=1) - 1
                    
                if atom_electrons and radical_electrons == 0:
                    radical_electrons = atom_electrons

                atom_map_num = self.encode_to_number((atom_hybridization, non_H_valence, radical_electrons, atom_electrons))

            else:
                atom_map_num = 0

            result_mol.GetAtomWithIdx(atom_idx).SetNumExplicitHs(explict_Hs)
            result_mol.GetAtomWithIdx(atom_idx).SetIsotope(atom_isotope)
            result_mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(atom_map_num)
            result_mol.GetAtomWithIdx(atom_idx).SetAtomicNum(0)

        for atom_idx in retain_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            
            if atom.GetAtomicNum() > 1:
                radical_electrons = atom.GetNumRadicalElectrons()
                if radical_electrons:
                    total_valence = round(sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()]))
                    explict_Hs = [x.GetSymbol() for x in atom.GetNeighbors()].count('H')
                    non_H_valence = total_valence - explict_Hs
                    atom_hybridization =  self.basic.get_atom_hybration(atom, self.basic.dict_to_tuple(atom_radical_electrons))
                    radical_electrons = atom.GetNumRadicalElectrons()
                    atom_electrons = atom_radical_electrons.get(atom_idx, 0)
                    
                    atom_map_num = self.encode_to_number((atom_hybridization, non_H_valence, radical_electrons, atom_electrons))
                    result_mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(atom_map_num)

        for atom in remove_atoms:
            result_mol.RemoveAtom(atom)

        bond_smi = Chem.MolToSmiles(result_mol)
        bond_smi = self.group_to_original_smi(bond_smi)
        bond_smi =  Chem.MolToSmiles(Chem.MolFromSmiles(bond_smi, sanitize=False))

        # Remove the chiral conformation for SMILES.
        bond_smi = bond_smi.replace('@', '')
        mole_smi = self.bond_smi_to_mole_smi(bond_smi, flag)

        bonds.append(bond_smi)
        smiles.append(mole_smi)
        
        return bonds, smiles

    @lru_cache(maxsize=128)
    def group_to_original_smi(self, bond_smi):
        mol = Chem.MolFromSmiles(bond_smi, sanitize=False)

        if Chem.MolToSmiles(mol).count('[') != bond_smi.count('[') and not any(x.islower() for x in bond_smi):
            mol = Chem.MolFromSmiles(bond_smi)

        result_mol = Chem.RWMol(mol)
        replace_atoms = []
        group_dict = Counter(self.para['charge_group']) + Counter(self.para['aromatic_group'])

        for atom in mol.GetAtoms():
            if atom.GetIsotope() in group_dict.values():
                replace_atoms.append(atom.GetIdx())
                group = Chem.MolFromSmiles({v: k for k, v in group_dict.items()}[atom.GetIsotope()])
                
                for x in group.GetAtoms():
                    result_mol.AddAtom(x)

                for x in group.GetBonds():
                    result_mol.AddBond(result_mol.GetNumAtoms() - x.GetBeginAtomIdx() - 1, result_mol.GetNumAtoms() - x.GetEndAtomIdx() - 1, x.GetBondType())

                for i, v in enumerate(atom.GetNeighbors()):
                    bond_type = mol.GetBondBetweenAtoms(atom.GetIdx(), v.GetIdx()).GetBondType()
                    result_mol.AddBond(result_mol.GetNumAtoms() - group.GetNumAtoms() + i, v.GetIdx(), bond_type)
                    if result_mol.GetBondBetweenAtoms(atom.GetIdx(),v.GetIdx()):
                        result_mol.RemoveBond(atom.GetIdx(), v.GetIdx())

        for atom in sorted(replace_atoms, reverse = True):
            result_mol.RemoveAtom(atom)

        smi = Chem.MolToSmiles(result_mol)

        return smi
    
    @lru_cache(maxsize=128)
    def add_virtual_bond(self, mol, result_mol, both_ends, atomic_num):
        if atomic_num == 0:
            atomic_num = 6
        result_mol.GetAtomWithIdx(result_mol.GetNumAtoms() - 1).SetIsotope(atomic_num)
        bond_type = mol.GetBondBetweenAtoms(both_ends[0], both_ends[1]).GetBondType()
        if not result_mol.GetBondBetweenAtoms(result_mol.GetNumAtoms() - 1, both_ends[0]):
            result_mol.AddBond(result_mol.GetNumAtoms() - 1, both_ends[0], bond_type)
        if result_mol.GetBondBetweenAtoms(both_ends[0], both_ends[1]):
            result_mol.RemoveBond(both_ends[0], both_ends[1])
        return result_mol
    
    @lru_cache(maxsize=128)
    def get_mBAC_bonds(self, smi):
        bonds = []
        cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
        mol = self.basic.get_mol_structure(std_smi, removeHs=False)

        for bond in mol.GetBonds():
            retain_atoms = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            bond_type = self.get_mBAC_bond(std_smi, mol, retain_atoms)
            bonds.append(f'{bond_type}')

        return Counter(bonds)

    @lru_cache(maxsize=128)
    def get_mBAC_bond(self, smi, mol, retain_atoms):
        aromatic_ring_size, atom_map_num = [], []
        atom_radical_electrons = self.basic.get_atom_radical_electrons(smi, AddHs=True)
        bond = mol.GetBondBetweenAtoms(*retain_atoms)
        
        if bond.GetBondTypeAsDouble() == 1.5:
            smallest_rings = (set(x) for x in Chem.GetSymmSSSR(mol))
            aromatic_atoms = set([x.GetIdx() for x  in mol.GetAromaticAtoms()])
            aromatic_rings = set([tuple(sorted(x)) for x in smallest_rings if set(x).issubset(aromatic_atoms)])
            for ring in aromatic_rings:
                if set(retain_atoms).issubset(ring):
                    aromatic_ring_size.append(len(ring))
        
        if len(aromatic_ring_size) == 2:
            bond_map_num = ''.join(map(str,sorted(aromatic_ring_size)))
        elif len(aromatic_ring_size) == 1:
            bond_map_num = f'0{aromatic_ring_size[0]}'
        else:
            bond_map_num = '00'

        for atom_idx in retain_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_isotope = atom.GetAtomicNum()
            connect_bonds = len([x for x in atom.GetBonds()])
            total_valence = round(sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()]))
            explict_Hs = [x.GetSymbol() for x in atom.GetNeighbors()].count('H')
            non_H_valence = total_valence - explict_Hs + abs(atom.GetFormalCharge())
            atom_hybridization =  str(atom.GetHybridization())[-1]
            atom_hybri_idx = int(atom_hybridization) if atom_hybridization.isdigit() else 1
            radical_electrons = atom.GetNumRadicalElectrons()
            atom_electrons = atom_radical_electrons.get(atom_idx, 0) + atom.GetFormalCharge()
            atom_map_num.append((atom_isotope, connect_bonds, non_H_valence, atom_hybri_idx, radical_electrons, atom_electrons))

        atom_map_num = [''.join(map(str, x)) for x in sorted(atom_map_num, reverse=True)]
        bond_type = f"{'_'.join(atom_map_num)}_{bond_map_num}"

        return bond_type

