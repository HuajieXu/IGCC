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

from igcc._basic import Basic
from igcc._CBH import CBH

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
        IGCC_vacant, IGCC_vacant_smiles, mBAC_vacant, mBAC_vacant_smiles = {}, [], {}, []

        if exists(f"{self.para['work_path']}/training_smiles.txt"):
            with open(f"{self.para['work_path']}/training_smiles.txt") as f:
                for x in f.readlines()[1:]:
                    if x.strip():
                        species, smi = x.strip().split()
                        cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
                        micro_mol.add(std_smi)
                        self.para['training_smiles'].add(std_smi)

        with open(f"{self.para['work_path']}/{self.para['input_file']}") as f:
            for x in tqdm(f.read().strip().split('\n')[1:], desc='Reading smiles information'):
                if x.strip():
                    species, smi = x.strip().split()
                    cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
                    formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(cal_smi))
                    smiles.append(std_smi)

                    reac, prod, CBH_reaction = self.CBH.get_CBH_reactions(std_smi, 3)

                    species_dict.setdefault('species', []).append(species)
                    species_dict.setdefault('inchikey', []).append(inc)
                    species_dict.setdefault('formula', []).append(formula)
                    species_dict.setdefault('smiles', []).append(std_smi)
                    species_dict.setdefault('standard_smiles', []).append(std_smi)
                    species_dict.setdefault('CBH_reaction', []).append(CBH_reaction)
                    
                    if reac != prod:
                        macro_mol.add(std_smi), micro_mol.update(set(reac + prod - Counter([std_smi])))

                        if self.para['check_mBAC_IGCC_smiles']:

                            # Process IGCC parameters.
                            IGCC_bonds = self.CBH.get_CBH_delta_bonds(std_smi, 3)[-1]

                            for IGCC_bond in IGCC_bonds:
                                if IGCC_bond not in self.para['IGCC_parameters']:

                                    IGCC_smi = self.CBH.bond_smi_to_mole_smi(IGCC_bond, 'high')

                                    if IGCC_smi:
                                        bonds = self.CBH.get_bonds_count(IGCC_smi, 3)

                                        if IGCC_bond == IGCC_smi or IGCC_bond in bonds:
                                            ref_cal_smi = self.basic.smi_to_std_format(IGCC_smi)[0]
                                            std_mol, ref_mol = Chem.MolFromSmiles(std_smi), Chem.MolFromSmiles(ref_cal_smi)
                                        
                                            if std_mol.GetNumHeavyAtoms() - ref_mol.GetNumHeavyAtoms() > 4:
                                                ref_smi = IGCC_smi
                                            else:
                                                ref_smi = std_smi
                                        else:
                                            ref_smi = std_smi
                                    else:
                                        ref_smi = std_smi
                                    IGCC_vacant.setdefault(IGCC_bond, set()).add(ref_smi)
                    
                    else:
                        micro_mol.add(std_smi)

                    if self.para['check_mBAC_IGCC_smiles']:
                        # Process mBAC parameters.
                        all_smiles = set(list(reac) + list(prod))
                        for smi in all_smiles:
                            cal_smi, std_smi, inc = self.basic.smi_to_std_format(smi)
                            mBAC_bonds = self.CBH.get_mBAC_bonds(smi)

                            for mBAC_bond in mBAC_bonds:
                                if mBAC_bond not in self.para['mBAC_parameters']:
                                    ref_smi = std_smi
                                    mBAC_vacant.setdefault(mBAC_bond, set()).add(ref_smi)
                                    print(mBAC_bond, std_smi)

        # To determine the referenced molecule for the IGCC paramters.
        for k, v in IGCC_vacant.items():
            ref_smi = sorted(v, key=lambda x: self.basic.get_all_atoms(x))[0]
            IGCC_vacant_smiles.append(ref_smi)

        # To determine the referenced molecule for the mBAC paramters.
        for k, v in mBAC_vacant.items():
            ref_smi = sorted(v, key=lambda x: self.basic.get_all_atoms(x))[0]
            mBAC_vacant_smiles.append(ref_smi)
        
        # To get all SMILES for the whole system.
        smiles = list(set([self.basic.smi_to_std_format(x)[1] for x in smiles + list(micro_mol) + list(macro_mol) + mBAC_vacant_smiles + IGCC_vacant_smiles]))
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
        
        # To process the vacant smiles for IGCC and mBAC parameters.
        if IGCC_vacant_smiles:
            print(f"\n{' Missing IGCC parameters! ':#^64}\n")
            IGCC_vacant_smiles = sorted(set(IGCC_vacant_smiles), key=lambda x: self.basic.get_all_atoms(x))
            
            with open(f"{self.para['work_path']}/IGCC_training.txt", 'w') as f:
                f.write(f"{'S/N':>6}{'Train_smiles':>64}\n")
                for i, v in enumerate(IGCC_vacant_smiles):
                    f.write(f'{i+1:>6}{v:>64}\n')
                    self.para['training_smiles'].add(v)

        else:
            if exists(f"{self.para['work_path']}/IGCC_training.txt"):
                remove(f"{self.para['work_path']}/IGCC_training.txt")

        if mBAC_vacant_smiles:
            print(f"\n{' Missing mBAC parameters! ':#^64}\n")
            mBAC_vacant_smiles = sorted(set(mBAC_vacant_smiles), key=lambda x: self.basic.get_all_atoms(x))
            with open(f"{self.para['work_path']}/mBAC_training.txt", 'w') as f:
                f.write(f"{'S/N':>6}{'Train_smiles':>64}\n")
                for i, v in enumerate(mBAC_vacant_smiles):
                    f.write(f'{i+1:>6}{v:>64}\n')
        else:
            if exists(f"{self.para['work_path']}/mBAC_training.txt"):
                remove(f"{self.para['work_path']}/mBAC_training.txt")


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

        self.para.update({'training_smiles':set()})
        
        if exists(f"{self.para['work_path']}/mBAC_parameters.txt"):
            with open(f"{self.para['work_path']}/{'mBAC_parameters.txt'}") as f:
                self.para.update({'mBAC_parameters': eval(f.read())})
        else:
            self.para.update({'mBAC_parameters':{}})

        if exists(f"{self.para['work_path']}/IGCC_parameters.txt"):
            with open(f"{self.para['work_path']}/{'IGCC_parameters.txt'}") as f:
                self.para.update({'IGCC_parameters': eval(f.read())})
        else:
            self.para.update({'IGCC_parameters':{}})

        if exists(f"{self.para['work_path']}/same_inchikey.txt"):
            with open(f"{self.para['work_path']}/same_inchikey.txt") as f:
                self.para.update({'same_inchikey': eval(f.read().replace(',\n', ','))})
        else:
            self.para.update({'same_inchikey':{}})

        if exists(f"{self.para['work_path']}/geometry_state.txt"):
            with open(f"{self.para['work_path']}/geometry_state.txt") as f:
                self.para.setdefault('geometry_state', {})
                for line in f.readlines()[1:]:
                    if line.strip():
                        _, inc, tar_smi, cur_smi, state = line.split()
                        if exists(f"{self.para['work_path']}/rawfiles/B3LYP/{inc}.out"):
                            self.para['geometry_state'].update({inc: [tar_smi, cur_smi, state]})
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

        if exists(f"{self.para['work_path']}/mBAC"):
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