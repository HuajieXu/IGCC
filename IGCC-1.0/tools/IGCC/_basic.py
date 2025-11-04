# -*- coding: utf-8 -*-
from re import findall
from math import isnan, ceil
from openbabel import pybel
from collections import Counter
from rdkit import Chem, RDLogger
from itertools import product, combinations
from functools import cmp_to_key, lru_cache
from rdkit.Chem import AllChem, rdDepictor, Descriptors, rdMolTransforms


rdDepictor.SetPreferCoordGen(True)
RDLogger.DisableLog('rdApp.*')

class Basic(object):
    """ To initiate parameters for Basic. """
    def __init__(self, para={'same_inchikey': {}}):
        self.para = para

    def list_set_to_tuple(self, lst_set):
        if isinstance(lst_set, list) or isinstance(lst_set, set):
            return tuple(self.list_set_to_tuple(item) for item in lst_set)
        else:
            return lst_set

    def dict_to_tuple(self, dct):
        if isinstance(dct, dict):
            return self.dict_to_tuple(tuple((k, v) for k, v in dct.items()))
        else:
            return dct

    def get_level_info(self, lst):
        level_info = {}
        lst_str = str(lst)
    
        for i, x in enumerate(lst_str):
            if x.isdigit():
                level = lst_str[:i].count('[') - lst_str[:i].count(']')
                level_info.setdefault(level - 1, []).append(int(x))
    
        return level_info

    def get_compared_lists(self, lst1, lst2):
        level_info1 = self.get_level_info(lst1)
        level_info2 = self.get_level_info(lst2)

        if level_info1 ==  level_info2:
            return 0

        for level in range(max(list(level_info1) + list(level_info2))):
            info1 = level_info1.get(level, [])
            info2 = level_info2.get(level, [])

            if max(info1, default=0) > max(info2, default=0):
                return 1
            elif max(info1, default=0) < max(info2, default=0):
                return -1
            else:
                while info1 or info2:
                    if info1.count(max(info1, default=0)) > info2.count(max(info2, default=0)):
                        return 1
                    elif info1.count(max(info1, default=0)) < info2.count(max(info2, default=0)):
                        return -1
                    else:
                        info1 = list(filter(lambda x: x != max(info1, default=0), info1))
                        info2 = list(filter(lambda x: x != max(info2, default=0), info2))

    def sorted_multidimension_lists(self, lst, reverse=False):
        if reverse:
            sorted_list = sorted(lst, key=cmp_to_key(self.get_compared_lists), reverse=True)
        else:
            sorted_list = sorted(lst, key=cmp_to_key(self.get_compared_lists), reverse=False)

        return sorted_list 


    @lru_cache(maxsize=128)
    def get_all_atoms(self, smi):
        cal_smi, _, _ = self.smi_to_std_format(smi)
        atoms  = Counter(z.GetSymbol() for z in Chem.AddHs(Chem.MolFromSmiles(cal_smi)).GetAtoms())
        
        return atoms.get('C', 0), atoms.get('H', 0), atoms.get('O', 0)


    @lru_cache(maxsize=128)
    def get_correct_cal_smi(self, ori_smi):
        cal_smi = ''
        
        cal_smi = Chem.MolToSmiles(Chem.MolFromSmiles(ori_smi))
        rdkit_aromaticity = True if findall('[a-z]', cal_smi) else False

        if rdkit_aromaticity:
            mol = pybel.readstring('smi', ori_smi)
            pybel_smi = mol.write('can').strip()
            
            pybel_aromaticity = True if findall('[a-z]', pybel_smi) else False

            if not pybel_aromaticity and pybel_smi:
                mol = Chem.MolFromSmiles(ori_smi, sanitize=False)
                cal_smi = Chem.MolToSmiles(mol)
            
            # Process the bug of radical aromaticity.                 
            if Chem.AddHs(Chem.MolFromSmiles(ori_smi)).GetNumAtoms() != Chem.AddHs(Chem.MolFromSmiles(cal_smi)).GetNumAtoms():
                cal_smi = Chem.MolToSmiles(Chem.MolFromSmiles(ori_smi))
        
        if not cal_smi:
            cal_smi = Chem.MolToSmiles(Chem.MolFromSmiles(ori_smi))

        return cal_smi


    @lru_cache(maxsize=128)
    def smi_to_std_format(self, smi, flag=True):
        if len(smi) > 2 and findall('^[CTZE]-', smi):
            try:
                mol = Chem.MolFromSmiles(smi)
                cal_smi = self.get_correct_cal_smi(smi[2:])
                std_smi = f'{smi[:2]}{cal_smi}'
                mol = Chem.MolFromSmiles(cal_smi)
                inc = f'{smi[:2]}{Chem.MolToInchiKey(mol)}-{Chem.Descriptors.NumRadicalElectrons(mol) + 1}'
            except:
                spin = findall('-(\d+)$', smi[2:])[-1]
                cal_smi = self.get_correct_cal_smi(findall('(.+)-\d+$', smi[2:])[-1])
                std_smi = f'{smi[:2]}{cal_smi}-{spin}'
                mol = Chem.MolFromSmiles(cal_smi)
                inc = f'{smi[:2]}{Chem.MolToInchiKey(mol)}-{spin}'

            finally:
                if flag and inc in self.para['same_inchikey']:
                    if std_smi not in self.para['same_inchikey'][inc]:
                        self.para['same_inchikey'][inc].append(std_smi)
                    
                    idx = self.para['same_inchikey'][inc].index(std_smi) + 1
                    inc = f'{inc[:29]}{idx}{inc[29:]}'
        
        else:
            try:
                mol = Chem.MolFromSmiles(smi)
                cal_smi = self.get_correct_cal_smi(smi)
                std_smi = f'{cal_smi}'
                mol = Chem.MolFromSmiles(cal_smi)
                inc = f'{Chem.MolToInchiKey(mol)}-{Chem.Descriptors.NumRadicalElectrons(mol) + 1}'

            except:
                spin = findall('-(\d+)$', smi)[-1]
                cal_smi = self.get_correct_cal_smi(findall('(.+)-\d+$', smi)[-1])
                std_smi = f'{cal_smi}-{spin}'
                mol = Chem.MolFromSmiles(cal_smi)
                inc = f'{Chem.MolToInchiKey(mol)}{smi[-2:]}'

            finally:
                if flag and inc in self.para['same_inchikey']:
                    if std_smi not in self.para['same_inchikey'][inc]:
                        self.para['same_inchikey'][inc].append(std_smi)
                    
                    idx = self.para['same_inchikey'][inc].index(std_smi) + 1
                    inc = f'{inc[:27]}{idx}{inc[27:]}'

        return cal_smi, std_smi, inc

   
    @lru_cache(maxsize=128)
    def get_mol_structure(self, smi, removeHs=True):
        cal_smi, std_smi, inc = self.smi_to_std_format(smi)

        mol = Chem.MolFromSmiles(cal_smi)

        if Chem.MolToSmiles(mol) != cal_smi:
            mol = Chem.MolFromSmiles(cal_smi, sanitize=False)
            
        rdDepictor.Compute2DCoords(mol)
        
        if removeHs == False:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed = 10)

        return mol


    @lru_cache(maxsize=128)
    def get_atom_radical_electrons(self, smi, AddHs=True):
        cal_smi, std_smi, inc = self.smi_to_std_format(smi)
        num_radical_electrons = int(findall('-(\d+$)', inc)[-1]) - 1

        if AddHs:
            mol = self.get_mol_structure(std_smi, removeHs=False)
        else:
            mol = self.get_mol_structure(std_smi, removeHs=True)

        atom_radical_electrons = {atom.GetIdx(): atom.GetNumRadicalElectrons() for atom in mol.GetAtoms() if atom.GetNumRadicalElectrons()} 
        total_radical_electrons = sum(atom_radical_electrons.values())

        if num_radical_electrons > total_radical_electrons:
            rings = [set(x) for x in Chem.GetSymmSSSR(mol) if len(set(x)) == 3]
            if rings:
                triple_bond = [y for x in rings for y in combinations(x, 2) if mol.GetBondBetweenAtoms(y[0], y[1]).GetBondTypeAsDouble() == 3]
                double_bond = [y for x in rings for y in combinations(x, 2) if mol.GetBondBetweenAtoms(y[0], y[1]).GetBondTypeAsDouble() == 2]
                if triple_bond:
                    atom_radical_electrons.update({x: 1 for x in triple_bond[0]})
                elif double_bond:
                    atom_radical_electrons.update({x: 1 for x in double_bond[0]})
                else:
                    print(f'!!! ERROR: The input spin multiplicity is not possible: --> {inc}: {std_smi}\nThe program exit abnormally.')
            else:
                for atom in mol.GetAtoms():
                    atom_mol = Chem.RWMol()
                    atom_mol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
                    atom_mol = atom_mol.GetMol()
                    valence_electrons = Descriptors.NumValenceElectrons(Chem.MolFromSmiles(f'[{atom.GetSymbol()}]'))
                    lone_pair_electrons = valence_electrons - atom.GetTotalValence()
                    if lone_pair_electrons:
                        atom_radical_electrons.update({atom.GetIdx(): atom_radical_electrons.get(atom.GetIdx(), 0) + lone_pair_electrons})

        charge_electrons_pair = [list(range(x, -2, -2)) for x in atom_radical_electrons.values()]
        all_charge_electrons_pair = set(product(*charge_electrons_pair))

        try:
            viable_charge_electrons_pair = [x for x in all_charge_electrons_pair if sum(x) == num_radical_electrons]
            viable_charge_electrons_pair = sorted(viable_charge_electrons_pair, reverse=True)[0]
        except:
            viable_charge_electrons_pair = []

        atom_radical_electrons = dict(zip(atom_radical_electrons, viable_charge_electrons_pair))

        return atom_radical_electrons


    def get_connect_group(self, atom, mol, all_idx):
        all_groups = [atom.GetAtomicNum()]
        all_idx.append(atom.GetIdx())
        connected_atoms = sorted([(x.GetAtomicNum(), x.GetIdx()) for x in atom.GetNeighbors() if x.GetIdx() not in all_idx], reverse=True)

        for x in connected_atoms:
            all_groups.append(self.get_connect_group(mol.GetAtomWithIdx(x[1]), mol, all_idx)[0])
  
        return all_groups, atom.GetIdx()

    @lru_cache(maxsize=128)
    def get_atom_hybration(self, atom, atom_radical_electrons):
        atom_radical_electrons = dict(atom_radical_electrons)
        if atom.GetIsotope() >= 100:
            hybridization = 4 - max([round(x.GetBondTypeAsDouble()) for x in atom.GetBonds()])

        else:
            try:
                total_valence = atom.GetTotalValence()
                total_degree = atom.GetTotalDegree()
            except:
                total_valence = ceil(sum([x.GetBondTypeAsDouble() for x in atom.GetBonds()]))
                total_degree = len([x for x in atom.GetBonds()])

            if atom.GetIdx() in atom_radical_electrons:
                unpaired_electrons = atom_radical_electrons.get(atom.GetIdx(), 0)
            else:
                unpaired_electrons = atom.GetNumRadicalElectrons()
            
            valence_electrons = Descriptors.NumValenceElectrons(Chem.MolFromSmiles(f'[{atom.GetSymbol()}]'))

            lone_pairs = (valence_electrons - total_valence - abs(unpaired_electrons)) // 2
            
            if unpaired_electrons and unpaired_electrons + sum([atom_radical_electrons.get(x.GetIdx(), 0) for x in atom.GetNeighbors()]) == 0:
                hybridization =  lone_pairs + total_degree - 1
            else:
                hybridization = abs(unpaired_electrons) + lone_pairs + total_degree - 1

            # Set the p-pi conjugated atom  hybration to SP2. 
            if hybridization == 3 and (unpaired_electrons > 0 or lone_pairs > 0):
                for bond in atom.GetBonds():
                    for adjacent_bond in [x for x in bond.GetOtherAtom(atom).GetBonds() if x.GetIdx() != bond.GetIdx()]:
                        if adjacent_bond.GetBondTypeAsDouble() > 1.0:
                            hybridization = 2
                            break
        return hybridization


    @lru_cache(maxsize=128)
    def get_configuration_type(self, smi, bond, ori_mol):
            cal_smi, std_smi, inc = self.smi_to_std_format(smi)
            atom_radical_electrons = self.get_atom_radical_electrons(smi, AddHs=True)
            
            mol = Chem.RWMol(ori_mol)
            left_groups, right_groups = [], []
            intersection = lambda a, b: [x for x in a if x in b]
            left_atom = bond.GetBeginAtom()
            right_atom = bond.GetEndAtom()

            if left_atom.GetDegree() <= 1 or right_atom.GetDegree() <= 1:
                return None, None, []

            elif self.get_atom_hybration(bond.GetBeginAtom(), self.dict_to_tuple(atom_radical_electrons)) <= 2 and self.get_atom_hybration(bond.GetEndAtom(), self.dict_to_tuple(atom_radical_electrons)) <= 2:
                left_heavy_atoms = [x for x in left_atom.GetNeighbors() if x.GetIdx() != right_atom.GetIdx()]
                right_heavy_atoms = [x for x in right_atom.GetNeighbors() if x.GetIdx() != left_atom.GetIdx()]

                left_bonds = [mol.GetBondBetweenAtoms(left_atom.GetIdx(), x.GetIdx()) for x in left_heavy_atoms]
                right_bonds = [mol.GetBondBetweenAtoms(right_atom.GetIdx(), x.GetIdx()) for x in right_heavy_atoms]
                left_groups, right_groups, left_non_intersection, right_non_intersection = [], [], [], []
                
                for x in left_heavy_atoms:
                    group = self.get_connect_group(x, mol, [right_atom.GetIdx()])
                    left_groups.append(group)
                for x in right_heavy_atoms:
                    group = self.get_connect_group(x, mol, [left_atom.GetIdx()])
                    right_groups.append(group)

                intersection = [x[0] for x in left_groups if x[0] in list(zip(*right_groups))[0]]
                for x in left_groups:
                    if x[0] not in left_non_intersection:
                        left_non_intersection.append(x[0])
                for x in right_groups:
                    if x[0] not in right_non_intersection:
                        right_non_intersection.append(x[0])

                if intersection and len(left_non_intersection) == len(left_groups) and len(right_non_intersection) == len(right_groups) and not len(left_groups) == len(right_groups) == 1:
                    left_group_idx = list(zip(*left_groups))[0].index(intersection[0])
                    left_group_atom = left_groups[left_group_idx][1]
                    right_group_idx = list(zip(*right_groups))[0].index(intersection[0])
                    right_group_atom = right_groups[right_group_idx][1]
                    try:
                        dihedral_angle = abs(rdMolTransforms.GetDihedralDeg(mol.GetConformer(), left_group_atom, left_atom.GetIdx(), right_atom.GetIdx(), right_group_atom))
                    except:
                        mol = Chem.RWMol(self.get_mol_structure(std_smi, removeHs=False))
                        dihedral_angle = abs(rdMolTransforms.GetDihedralDeg(mol.GetConformer(), left_group_atom, left_atom.GetIdx(), right_atom.GetIdx(), right_group_atom))
                    
                    if isnan(dihedral_angle):
                        return None, None, []
                    
                    tag = 'C' if dihedral_angle < 90 else 'T'
                    
                    return tag, dihedral_angle, [left_group_atom, left_atom.GetIdx(), right_atom.GetIdx(), right_group_atom]
                
                elif not intersection and len(left_non_intersection) == len(left_groups) and len(right_non_intersection) == len(right_groups) and not len(left_groups) == len(right_groups) == 1:
                    left_group_atom = left_groups[0][1]
                    right_group_atom = right_groups[0][1]
                    
                    # Process the bug of bad conformer.
                    try:
                        dihedral_angle = abs(rdMolTransforms.GetDihedralDeg(mol.GetConformer(), left_group_atom, left_atom.GetIdx(), right_atom.GetIdx(), right_group_atom))
                    except:
                        mol = Chem.RWMol(self.get_mol_structure(std_smi, removeHs=False))
                        dihedral_angle = abs(rdMolTransforms.GetDihedralDeg(mol.GetConformer(), left_group_atom, left_atom.GetIdx(), right_atom.GetIdx(), right_group_atom))
                    
                    if isnan(dihedral_angle):
                        return None, None, []
                    
                    tag = 'Z' if dihedral_angle < 90 else 'E'
                    
                    return tag, dihedral_angle, [left_group_atom, left_atom.GetIdx(), right_atom.GetIdx(), right_group_atom]
                
                else:
                    return None, None, []
            else:
                return None, None, []
