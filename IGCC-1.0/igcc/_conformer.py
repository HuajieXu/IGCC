# -*- coding: utf-8 -*-
from re import findall
from rdkit import Chem
from functools import cmp_to_key
from openbabel import openbabel, pybel
from rdkit.Chem import AllChem, rdMolTransforms

from igcc._basic import Basic

pybel.ob.obErrorLog.StopLogging()

class Conformer(object):

    """ To initiate parameters for Conformer. """
    def __init__(self, inc, para, structure_method = 1):
        self.inc = inc
        self.para = para
        self.basic = Basic(para)
        self.structure_method = structure_method

    """ To generate the gjf file from rdkit. """
    def rdkit_to_molblock(self, smi):
        cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
        spin = int(findall('-(\d+$)', inc)[-1])

        mol = Chem.AddHs(Chem.MolFromSmiles(cal_smi))
        AllChem.EmbedMolecule(mol, randomSeed = 10)
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except:
            pass
        molblock = Chem.MolToMolBlock(mol)
        
        molblock = self.get_mol_configuration(std_smi, spin, molblock)

        return molblock



    """ Generate the gjf file from openbabel. """
    def pybel_to_molblock(self, smi):
        cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
        spin = int(findall('-(\d+$)', inc)[-1])

        mol = pybel.readstring('smi', cal_smi)
        
        #There is a bug when constructing 3D by gen3d method. 
        if cal_smi.count('/') < 2:
            gen3d = openbabel.OBOp.FindType("gen3D")
            gen3d.Do(mol.OBMol, "--best")
        else:
            mol.make3D()

        molblock = mol.write('mol')

        # Check if exist the bug of transfinite coordinates.
        if '*' in molblock:
            molblock = rdkit_to_molblock(cal_smi)

        molblock = self.get_mol_configuration(std_smi, spin, molblock)

        return molblock


    """ Generate the specific configuration of molecules. """
    def get_mol_configuration(self, smi, spin, molblock):
        cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)

        if findall('^[CTZE]-', std_smi):
            conf_bond = []
            mol =  Chem.MolFromMolBlock(molblock, removeHs=False)
            for bond in mol.GetBonds():
                conf_type, dihedral_angle, atoms = self.basic.get_configuration_type(std_smi, bond, mol)
                if conf_type:
                    conf_bond.append([conf_type, dihedral_angle, atoms])

            if len(conf_bond) == 1 and 'CTZE'.index(conf_bond[0][0]) % 2 != 'CTZE'.index(std_smi[0]) % 2:
                try:
                    rdMolTransforms.SetDihedralDeg(mol.GetConformer(), *conf_bond[0][-1], abs(conf_bond[0][1] - 180))
                except:
                    pass
            
            elif len(conf_bond) > 1 and std_smi[0] not in list(zip(*conf_bond))[0]:
                groups = []
                
                for i, x in enumerate(conf_bond):
                    left_group = self.basic.get_connect_group(mol.GetAtomWithIdx(x[2][0]), mol, [x[2][1]])[0]
                    right_group = self.basic.get_connect_group(mol.GetAtomWithIdx(x[2][3]), mol, [x[2][2]])[0]
                    groups.append(left_group + right_group)

                sorted_groups = self.basic.sorted_multidimension_lists(groups, reverse=True)
                aim_idx = [i for i, v in enumerate(groups) if sorted_groups[0] == v][0]
           
                try:
                    rdMolTransforms.SetDihedralDeg(mol.GetConformer(), *conf_bond[aim_idx][-1], abs(conf_bond[aim_idx][1] - 180))
                except:
                    pass
            
            molblock = Chem.MolToMolBlock(mol)
        
        return molblock



    """ To generate the standard gjf file for Gaussian. """
    def output_standard_gjf(self, smi=None):
        if not smi:
            smi  = self.para['species'][self.inc]

        cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
        mol = Chem.AddHs(Chem.MolFromSmiles(cal_smi))

        spin = findall('-(\d+$)', inc)[-1]
        cal_spin = sum(x.GetNumRadicalElectrons() for x in Chem.MolFromSmiles(cal_smi).GetAtoms()) + 1
        N_heavyatoms = max(mol.GetNumHeavyAtoms(), 1)
        memory = N_heavyatoms * self.para['number_process'] * self.para['B3LYP_mem_per_core']

        # Add additional keywords to Multiple free radicals.
        if len(findall('\[', smi)) > 1 and int(spin) > 2 or cal_spin < int(spin):
            additional_keywords = 'guess=mix'
        else:
            additional_keywords = ''

        # Get 3D structure.
        if self.structure_method == 2:
            molblock = self.rdkit_to_molblock(std_smi)
        else:
            molblock = self.pybel_to_molblock(std_smi)

        mol = pybel.readstring('mol', molblock)
        coord = '\n'.join(mol.write('xyz').split('\n')[2:-1])

        # Produce full format.
        gjf = f"%nprocshared={self.para['number_process']}\n%mem={memory}MB\n{self.para['key_words']} "\
            f"scale={self.para['scale_factor']['fund']} press={self.para['scale_factor']['press']} {additional_keywords}"\
            f"\n\n{smi}_{self.structure_method}\n\n0 {spin}\n{coord}\n\n"\

        return gjf
