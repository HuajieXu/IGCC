# -*- coding: utf-8 -*-
from glob import glob
from tqdm import tqdm
from rdkit import Chem
from re import findall
from os.path import exists
from itertools import chain
from os import makedirs, remove
from shutil import copy, rmtree 
from collections import Counter
from functools import lru_cache
from pandas import DataFrame, read_csv
from rdkit.Chem import rdMolDescriptors
from configparser import RawConfigParser

from tdoc._basic import Basic
from tdoc._CBH import CBH

import logging

logging.basicConfig(level=logging.ERROR)

class Parameters():
    
    """ To initiate parameters for Parameters. """
    def __init__(self, input_file, para_file, work_path):
        self.para={'input_file': input_file, 'para_file': para_file, 'work_path': work_path}


    """ To get SMILES. """
    @lru_cache(maxsize=128)
    def get_smiles(self):
        smiles, species_dict, micro_mol, macro_mol = [], {}, set(), set()
        BDC_vacant, BDC_vacant_smiles, BAC_vacant, BAC_vacant_smiles = {}, [], {}, []

        if exists(f"{self.para['work_path']}/given_micro_smiles.txt"):
            with open(f"{self.para['work_path']}/given_micro_smiles.txt") as f:
                for x in f.readlines()[1:]:
                    if x.strip():
                        species, smi = x.strip().split()
                        cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
                        micro_mol.add(std_smi)
                        self.para['given_micro_smiles'].add(std_smi)

        with open(f"{self.para['work_path']}/{self.para['input_file']}") as f:
            for x in tqdm(f.read().strip().split('\n')[1:], desc='Reading smiles information'):
                if x.strip():
                    species, smi = x.strip().split()
                    cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
                    formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(cal_smi))
                    smiles.append(std_smi)

                    reac, prod, IGCC_reaction = self.CBH.get_CBH_reactions(std_smi, 3)

                    species_dict.setdefault('species', []).append(species)
                    species_dict.setdefault('inchikey', []).append(inc)
                    species_dict.setdefault('formula', []).append(formula)
                    species_dict.setdefault('smiles', []).append(std_smi)
                    species_dict.setdefault('standard_smiles', []).append(std_smi)
                    species_dict.setdefault('IGCC_reaction', []).append(IGCC_reaction)
                    
                    if reac != prod:
                        macro_mol.add(std_smi), micro_mol.update(set(reac + prod - Counter([std_smi])))

                        if self.para['check_BAC_BDC_smiles']:

                            # Process BDC parameters.
                            BDC_bonds = self.CBH.get_CBH_delta_bonds(std_smi, 3)[-1]

                            for BDC_bond in BDC_bonds:
                                if BDC_bond not in self.para['BDC_parameters']:

                                    BDC_smi = self.CBH.bond_smi_to_mole_smi(BDC_bond, 'high')

                                    if BDC_smi:
                                        bonds = self.CBH.get_bonds_count(BDC_smi, 3)

                                        if BDC_bond == BDC_smi or BDC_bond in bonds:
                                            ref_cal_smi = self.basic.smi_to_std_format(BDC_smi)[0]
                                            std_mol, ref_mol = Chem.MolFromSmiles(std_smi), Chem.MolFromSmiles(ref_cal_smi)
                                        
                                            if std_mol.GetNumHeavyAtoms() - ref_mol.GetNumHeavyAtoms() > 4:
                                                ref_smi = BDC_smi
                                            else:
                                                ref_smi = std_smi
                                        else:
                                            ref_smi = std_smi
                                    else:
                                        ref_smi = std_smi
                                    BDC_vacant.setdefault(BDC_bond, set()).add(ref_smi)
                    
                    else:
                        micro_mol.add(std_smi)

                    if self.para['check_BAC_BDC_smiles']:
                        # Process BAC parameters.
                        all_smiles = set(list(reac) + list(prod))
                        for smi in all_smiles:
                            cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
                            BAC_bonds = self.CBH.get_BAC_bonds(smi)

                            for BAC_bond in BAC_bonds:
                                if BAC_bond not in self.para['BAC_parameters']:
                                    ref_smi = std_smi
                                    BAC_vacant.setdefault(BAC_bond, set()).add(ref_smi)
                                    print(BAC_bond, std_smi)

        # To determine the referenced molecule for the BDC paramters.
        for k, v in BDC_vacant.items():
            ref_smi = sorted(v, key=lambda x: self.basic.get_all_atoms(x))[0]
            BDC_vacant_smiles.append(ref_smi)

        # To determine the referenced molecule for the BAC paramters.
        for k, v in BAC_vacant.items():
            ref_smi = sorted(v, key=lambda x: self.basic.get_all_atoms(x))[0]
            BAC_vacant_smiles.append(ref_smi)
        
        # To get all SMILES for the whole system.
        smiles = list(set([self.basic.smi_to_std_format(x)[1] for x in smiles + list(micro_mol) + list(macro_mol) + BAC_vacant_smiles + BDC_vacant_smiles]))
        smiles = sorted(smiles, key=lambda x: self.basic.get_all_atoms(x))

        # To sort all data list by inchikeys.
        data_list, species_list = [], []
        for smi in smiles:
            cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
            data_list.append([inc, std_smi])
            species_list.append([std_smi, inc])

        # To check whether there are duplicate inchikeys.
        duplicate_inchikey = {}
        df = DataFrame(species_list)
        dup = df.groupby(df.columns[1]).filter(lambda x: len(x) > 1).groupby(1)[0]
        for x, y in list(dup):
            duplicate_inchikey.update({x: sorted(y.values.tolist())})

        if set(chain(*duplicate_inchikey.values())) - set(chain(*self.para['same_inchikey'].values())):
            for k, v in duplicate_inchikey.items():
                for x in v:
                    if x not in self.para['same_inchikey'].get(k, []):
                        self.para['same_inchikey'].setdefault(k, []).append(x)

            with open(f"{self.para['work_path']}/same_inchikey.txt", 'w') as f:
                f.write(str(self.para['same_inchikey']).replace('],', '],\n'))

        data_list = sorted(data_list)

        for inc, std_smi in data_list:
            self.para.setdefault('species', {}).update({inc: std_smi})
            self.para.setdefault('macro_mol', [])
            self.para.setdefault('micro_mol', [])
            if std_smi in micro_mol:
                self.para['micro_mol'].append(inc)
            if std_smi in macro_mol:
                self.para['macro_mol'].append(inc)

        DataFrame(species_dict).to_csv(f"{format(self.para['work_path'])}/csvfiles/input_data.csv", index=False)
        DataFrame(list(self.para['species'].items())).to_csv(f"{format(self.para['work_path'])}/csvfiles/all_smiles.csv", index=False, header = ['inchikey', 'standard_smiles'])
        
        # To process the vacant smiles for BDC and BAC parameters.
        if BDC_vacant_smiles:
            print(f"\n{' Missing BDC parameters! ':#^64}\n")
            BDC_vacant_smiles = sorted(set(BDC_vacant_smiles), key=lambda x: self.basic.get_all_atoms(x))
            
            with open(f"{self.para['work_path']}/BDC_training.txt", 'w') as f:
                f.write(f"{'S/N':>6}{'Train_smiles':>64}\n")
                for i, v in enumerate(BDC_vacant_smiles):
                    f.write(f'{i+1:>6}{v:>64}\n')
                    self.para['given_micro_smiles'].add(v)

        else:
            if exists(f"{self.para['work_path']}/BDC_training.txt"):
                remove(f"{self.para['work_path']}/BDC_training.txt")

        if BAC_vacant_smiles:
            print(f"\n{' Missing BAC parameters! ':#^64}\n")
            BAC_vacant_smiles = sorted(set(BAC_vacant_smiles), key=lambda x: self.basic.get_all_atoms(x))
            with open(f"{self.para['work_path']}/BAC_training.txt", 'w') as f:
                f.write(f"{'S/N':>6}{'Train_smiles':>64}\n")
                for i, v in enumerate(BAC_vacant_smiles):
                    f.write(f'{i+1:>6}{v:>64}\n')
        else:
            if exists(f"{self.para['work_path']}/BAC_training.txt"):
                remove(f"{self.para['work_path']}/BAC_training.txt")


    """ To get input parameters. """
    @lru_cache(maxsize=128)
    def get_input_parameters(self):
        
        # Process the Capitalization issue.
        config = RawConfigParser()
        config.optionxform = lambda option: option
        
        config.read(f"{self.para['work_path']}/{self.para['para_file']}")
        self.para.update({k: eval(v) for k, v in config.items('input_parameters')})
        self.para.update({k: eval(v) for k, v in config.items('server_parameters')})
        self.para.update({k: eval(v) for k, v in config.items('default_parameters')})

        self.para.update({'given_micro_smiles':set()})
        
        if exists(f"{self.para['work_path']}/BAC_parameters.txt"):
            with open(f"{self.para['work_path']}/{'BAC_parameters.txt'}") as f:
                self.para.update({'BAC_parameters': eval(f.read())})
        else:
            self.para.update({'BAC_parameters':{}})

        if exists(f"{self.para['work_path']}/BDC_parameters.txt"):
            with open(f"{self.para['work_path']}/{'BDC_parameters.txt'}") as f:
                self.para.update({'BDC_parameters': eval(f.read())})
        else:
            self.para.update({'BDC_parameters':{}})

        if exists(f"{self.para['work_path']}/same_inchikey.txt"):
            with open(f"{self.para['work_path']}/same_inchikey.txt") as f:
                self.para.update({'same_inchikey': eval(f.read().replace(',\n', ','))})
        else:
            self.para.update({'same_inchikey':{}})

        if exists(f"{self.para['work_path']}/geometry_state.txt"):
            with open(f"{self.para['work_path']}/geometry_state.txt") as f:
                for line in f.readlines()[1:]:
                    if line.strip():
                        _, inc, tar_smi, cur_smi, state = line.split()
                        if exists(f"{self.para['work_path']}/rawfiles/B3LYP/{inc}.out"):
                            self.para.setdefault('geometry_state', {}).update({inc: [tar_smi, cur_smi, state]})
        else:
            self.para.update({'geometry_state':{}})

        self.basic = Basic(self.para)
        self.CBH = CBH(self.para)

        

    """ To build working directories. """
    @lru_cache(maxsize=128)
    def build_work_dir(self):
        for x in ['B3LYP', 'GFNFF', 'MP2', 'CCSDT']:
            if not exists(f"{self.para['work_path']}/rawfiles/{x}"):
                makedirs(f"{self.para['work_path']}/rawfiles/{x}")
        
        for x in ['B3LYP', 'MP2', 'CCSDT']:
            if not exists(f"{self.para['work_path']}/datfiles/{x}"):
                makedirs(f"{self.para['work_path']}/datfiles/{x}")
        
        if not exists(f"{self.para['work_path']}/csvfiles"):
            makedirs(f"{self.para['work_path']}/csvfiles")
        
        if exists(f"{self.para['work_path']}/subfiles"):
            rmtree(f"{self.para['work_path']}/subfiles", True)

        if exists(f"{self.para['work_path']}/BAC"):
            rmtree(f"{self.para['work_path']}/subfiles", True)

        if exists(f"{self.para['work_path']}/subfiles"):
            rmtree(f"{self.para['work_path']}/subfiles", True)


    """ To get all parameters. """     
    def get_all_parameters(self):
        print('\nRead input parameters ...\n')
        self.get_input_parameters()
        self.build_work_dir()
        self.get_smiles()
        return self.para