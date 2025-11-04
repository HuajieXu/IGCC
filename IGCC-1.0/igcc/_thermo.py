# -*- coding: utf-8 -*-
from math import ceil
from rdkit import Chem
from re import findall
from itertools import chain
from os import chdir, listdir
from collections import Counter
from rdkit.Chem import Descriptors
from pandas import DataFrame, read_csv

import numpy as np

from igcc._CBH import CBH
from igcc._basic import Basic
from igcc._thermofit import Thermofit



class Thermo(object):

    """ To initiate parameters for Thermo. """
    def __init__(self, para):
        self.para = para
        self.basic = Basic(para)
        self.CBH = CBH(para)

    """ To calculate thermodymamic data for fitting thermodynamic paramters. """
    def get_data_from_out(self):
        data, unfinish = {}, []
        for inc in self.para['species']:
            for method in ['B3LYP', 'GFNFF', 'MP2', 'CCSDT']:
                if f'{inc}.out' in listdir(f"{self.para['work_path']}/rawfiles/{method}"):
                    data.setdefault(inc, {}).update(eval(f"self.get_{method}_energy_from_out('{inc}')"))

        for inc in self.para['species']:
            data.setdefault(inc, {}).update(self.get_final_thermo_data(data, inc))

        for inc in self.para['species']:
            if inc in self.para['macro_mol']:
                if data[inc].get('Hf_CCSDT_CBH/(kcal/mol)', None) == None:
                    unfinish.append(inc)
            else:
                if data[inc].get('Hf_CCSDT/(kcal/mol)', None) == None:
                    unfinish.append(inc)

        if unfinish:
            print(f"\n**** Thermodynamic calculation does not completed! ****\n\n{chr(10).join(' '.join(f'{y:30}' for y in x) for x in [unfinish[i*2:i*2+2] for i in range(len(unfinish)//2+1)])}\n")
        
        df = DataFrame(data).T
        input_df = read_csv(f"{self.para['work_path']}/csvfiles/input_data.csv", index_col = 0)
        
        df.to_csv(f"{self.para['work_path']}/csvfiles/source_data.csv", index_label = 'inchikey')
        df_index = list(df.columns)
        
        # Skip the state of without any data
        if 'standard_smiles' not in df.columns:
            return None
        
        if df_index[-8] == 'Hf_mBAC/(kcal/mol)':
            thermo_df = input_df.merge(df, on='standard_smiles').loc[:, input_df.columns[0:5].append(df.columns[-8:])]
        else:
            thermo_df = input_df.merge(df, on='standard_smiles').loc[:, input_df.columns[0:5].append(df.columns[-15:-7])]

        try:
            thermo_df['Cp/(cal/mol/K)'] = thermo_df['Cp/(cal/mol/K)'].map(lambda x: x.get(298.15))
        except:
            pass
        thermo_df = thermo_df.rename(columns = {'Cp/(cal/mol/K)': 'Cp_298/(cal/mol/K)'})
        thermo_df.to_csv(f"{self.para['work_path']}/csvfiles/thermo_data.csv", index_label = 'index')

        return None



    def get_final_thermo_data(self, data, inc):
        cal_smi =  self.basic.smi_to_std_format(self.para['species'][inc])[0]
        atoms = Counter(x.GetSymbol() for x in Chem.AddHs(Chem.MolFromSmiles(cal_smi)).GetAtoms())
        Exp_SO = round((-0.14 / 1000) * len(findall('C|c', self.para['species'][inc])) + (-0.36 / 1000) * len(findall('O|o', self.para['species'][inc])), 7)

        Hf_mBAC = self.get_mBAC_from_formula(inc)
        Hf_mBAC = round(Hf_mBAC, 2) if Hf_mBAC else Hf_mBAC

        Hf_IGCC = self.get_IGCC_from_formula(inc)
        Hf_IGCC = round(Hf_IGCC, 2) if Hf_IGCC else Hf_IGCC

        try:
            H_mol_BYP = data[inc]['E_BYP/(Eh)'] + data[inc]['H_corr/(Eh)'] + (self.para['scale_factor']['zpe'] 
                    - self.para['scale_factor']['fund']) * data[inc]['ZPE/(Eh)'] + Exp_SO
            Hf_B3LYP = 627.5095 * (H_mol_BYP - sum(v * self.para['h_atoms']['CCSDT'][k] for k, v in atoms.items()) 
                    + sum(v * self.para['eof_atoms'][k] for k, v in atoms.items())) + data[inc]['Hf_conf_298/(kcal/mol)']
            Hf_B3LYP = round(Hf_B3LYP, 2)
        except:
            Hf_B3LYP = None

        try:        
            H_mol_MP2 = data[inc]['E_MP2/(Eh)'] + data[inc]['H_corr/(Eh)'] + (self.para['scale_factor']['zpe'] 
                    - self.para['scale_factor']['fund']) * data[inc]['ZPE/(Eh)'] + Exp_SO
            Hf_MP2 = 627.5095 * (H_mol_MP2 - sum(v * self.para['h_atoms']['CCSDT'][k] for k, v in atoms.items()) 
                + sum(v * self.para['eof_atoms'][k] for k, v in atoms.items())) + data[inc]['Hf_conf_298/(kcal/mol)']
            Hf_MP2 = round(Hf_MP2, 2)
        except:
            Hf_MP2 = None 
     
        try:
            H_mol = round(data[inc]['CC_CBS/(Eh)'] + data[inc]['H_corr/(Eh)'] + (self.para['scale_factor']['zpe'] 
                - self.para['scale_factor']['fund']) * data[inc]['ZPE/(Eh)'] + Exp_SO, 7)
            Hf_CCSDT = 627.5095 * (H_mol - sum(v * self.para['h_atoms']['CCSDT'][k] for k, v in atoms.items()) 
                + sum(v * self.para['eof_atoms'][k] for k, v in atoms.items())) + data[inc]['Hf_conf_298/(kcal/mol)'] + Hf_mBAC
            Hf_CCSDT = round(Hf_CCSDT, 2) if Hf_CCSDT else Hf_CCSDT
        except:
            Hf_CCSDT = None

        if inc in self.para['macro_mol']:
            reac, prod, _ = self.CBH.get_CBH_reactions(self.para['species'][inc], 3)
            species = {v: k for k, v in self.para['species'].items()}

            try:
                E_reac = sum(v * (data[species[k]]['CC_CBS/(Eh)'] - data[species[k]]['E_MP2/(Eh)']) for k, v in (reac - Counter([self.para['species'][inc]])).items())
                E_prod = sum(v * (data[species[k]]['CC_CBS/(Eh)'] - data[species[k]]['E_MP2/(Eh)']) for k, v in prod.items())
                H_mol = round(data[inc]['E_MP2/(Eh)'] + E_prod - E_reac + data[inc]['H_corr/(Eh)'] + (self.para['scale_factor']['zpe'] 
                    - self.para['scale_factor']['fund']) * data[inc]['ZPE/(Eh)'] + Exp_SO, 7)
                
                Hf_CCSDT_CBH = 627.5095 * (H_mol - sum(v * self.para['h_atoms']['CCSDT'][k] for k, v in atoms.items()) 
                            + sum(v * self.para['eof_atoms'][k] for k, v in atoms.items())) + data[inc]['Hf_conf_298/(kcal/mol)'] + Hf_mBAC + Hf_IGCC
                Hf_CCSDT_CBH = round(Hf_CCSDT_CBH, 2) if Hf_CCSDT_CBH else Hf_CCSDT_CBH
            except:
                Hf_CCSDT_CBH = None
        else:
            Hf_CCSDT_CBH = None
            
        try:
            S_298 = data[inc]['S_RRHO_298/(cal/mol/K)'] + data[inc]['S_conf_298/(cal/mol/K)']
            Cp = {k: round(v + data[inc]['Cp_conf/(cal/mol/K)'][k], 2) for k, v in data[inc]['Cp_RRHO/(cal/mol/K)'].items()}
        except:
            S_298, Cp = None, None
        
        return {'Exp_SO/(Eh)': Exp_SO, 'Hf_mBAC/(kcal/mol)': Hf_mBAC, 'Hf_IGCC/(kcal/mol)': Hf_IGCC, 'Hf_B3LYP/(kcal/mol)': Hf_B3LYP, 'Hf_MP2/(kcal/mol)': Hf_MP2, 
                'Hf_CCSDT/(kcal/mol)': Hf_CCSDT, 'Hf_CCSDT_CBH/(kcal/mol)': Hf_CCSDT_CBH, 'S_298/(cal/mol/K)': S_298, 'Cp/(cal/mol/K)': Cp}



    """ To get B3LYP energy. """
    def get_B3LYP_energy_from_out(self, inc):
        with open(f"{self.para['work_path']}/rawfiles/B3LYP/{inc}.out") as f:
            content = f.read()
        E_BYP, H_BYP = np.round(np.array(findall('\WHF=(-\d*\.\d+).+?HTot=(-\d*\.\d+)', content.replace('\n ', ''))[-1], float), 7)
        try:
            ZPE = round(float(findall('ZeroPoint=(\d*\.\d+)', content.replace('\n ', ''))[-1]), 7)
        except:
            ZPE = 0
        H_corr = round(H_BYP - E_BYP, 7)
        Cv_RRHO_298, S_RRHO_298 = np.round(np.array(findall('\n Total\s+\d.+?(\S+)\s+(\S+)\n', content)[-1], float), 2)
        Freqs = np.array(''.join(findall(' Frequencies -- (.+?\n)', content)).split(), float)

        # Get Cp from Harmonic-Oscillator (HO) approximation.
        h = 6.62606957 * 10 ** -34
        kB = 1.3806488 * 10 ** -23
        c = 2.99792458 * 10 ** 10
        v = c * Freqs * self.para['scale_factor']['fund']
        func_Cv_vib = lambda T: sum(1.987 * np.exp(h * v / kB / T) * (h * v / kB / T / (np.exp(h * v / kB / T) - 1)) ** 2)
        Tlist = [100, 150, 200, 250, 298.15] + list(range(300, 1000, 50)) + list(range(1000, 2000, 100)) + list(range(2000, 5001, 250))
        Cp_RRHO = {T: round(func_Cv_vib(T) - func_Cv_vib(298.15) + Cv_RRHO_298 + 1.987, 2) for T in Tlist}
        Freqs = Freqs.tolist()
        return {'standard_smiles': self.para['species'][inc], 'E_BYP/(Eh)': E_BYP, 'ZPE/(Eh)': ZPE, 'H_corr/(Eh)': H_corr,
                'S_RRHO_298/(cal/mol/K)': S_RRHO_298, 'Cp_RRHO/(cal/mol/K)': Cp_RRHO, 'Freqs/(cm-1)': Freqs}



    """ To get CCSDT energy. """
    def get_CCSDT_energy_from_out(self, inc):
        with open(f"{self.para['work_path']}/rawfiles/CCSDT/{inc}.out") as f:
            content = f.read().replace('\n ', '')
        HF_DZ, CC_DZ, HF_TZ, CC_TZ = np.array(findall('\WHF=(-\d*\.\d+).+?CCSD\(T\)=(-\d*\.\d+)', content), dtype = float).flatten()
        HF_CBS = (HF_TZ * 3 ** 3.4 - HF_DZ * 2 ** 3.4) / (3 ** 3.4 - 2 ** 3.4)
        Corr_CBS = ((CC_TZ - HF_TZ) * 3 ** 2.4 - (CC_DZ - HF_DZ) * 2 ** 2.4) / (3 ** 2.4 - 2 ** 2.4)
        CC_CBS = HF_CBS + Corr_CBS
        HF_DZ, HF_TZ, CC_DZ, CC_TZ, HF_CBS, Corr_CBS, CC_CBS = map(lambda x: round(x, 7), [HF_DZ, HF_TZ, CC_DZ, CC_TZ, HF_CBS, Corr_CBS, CC_CBS])
        return {'HF_DZ/(Eh)': HF_DZ, 'HF_TZ/(Eh)': HF_TZ, 'CC_DZ/(Eh)': CC_DZ, 'CC_TZ/(Eh)': CC_TZ, 'HF_CBS/(Eh)': HF_CBS, 'Corr_CBS/(Eh)': Corr_CBS, 'CC_CBS/(Eh)': CC_CBS}



    """ To MP2 energy. """
    def get_MP2_energy_from_out(self, inc):
        with open(f"{self.para['work_path']}/rawfiles/MP2/{inc}.out") as f: content = f.read()
        SPE = float(findall('MP2=(-*\w*\.\w*)', (content.replace('\n ', '')))[-1])
        return {'E_MP2/(Eh)': SPE}



    """ To obtain conformational corrections. """
    def get_GFNFF_energy_from_out(self, inc):
        Tlist = [100, 150, 200, 250, 298.15] + list(range(300, 1000, 50)) + list(range(1000, 2000, 100)) + list(range(2000, 5001, 250))
        with open(f"{self.para['work_path']}/rawfiles/GFNFF/{inc}.out", encoding='utf-8') as f:
            content, E, g, p = f.read(), [], [], []
        
        try:
            aim_content = findall('origin\n([\s\S]+?)\nT', content)[-1]
        except:
            return {'Hf_conf_298/(kcal/mol)': 0.000, 'S_conf_298/(cal/mol/K)': 0.000, 'Cp_conf/(cal/mol/K)': dict(zip(Tlist, [0.000] * len(Tlist)))}

        for x in [x.split() for x in aim_content.split('\n')]:
            if len(x) > 5:
                E.append(x[1]), g.append(x[-2])
            p.append(x[3])
        
        E, g, T = np.array(E, float) * 1000 , np.array(g, int), np.array(Tlist, float)
        T = T.reshape(len(T), 1)
        E_beta = E / (1.987 * T)
    
        Hf_conf = float(findall('H\(T\)-H\(0\)\s+=\s+(\S+)', content)[-1])
        S_conf = float(findall('Sconf\s+=\s+(\S+)', content)[-1])
        Cp_298 = float(findall('Cp\(total\)\s+=\s+(\S+)', content)[-1])
        Cp_conf = 1.987 * np.dot(E_beta ** 2 * np.exp(- E_beta), g) / np.dot(np.exp(- E_beta), g) - 1.987 * (np.dot(E_beta * np.exp(- E_beta), g) / np.dot(np.exp(-E_beta), g)) ** 2
    
        if Cp_298:
            Cp_conf = Cp_conf * Cp_298 / Cp_conf[4]
    
        Cp_conf = dict(zip(Tlist, np.round(Cp_conf, 3)))

        return {'Hf_conf_298/(kcal/mol)': round(Hf_conf, 3), 'S_conf_298/(cal/mol/K)': round(S_conf, 3), 'Cp_conf/(cal/mol/K)': Cp_conf}


    """ To get mBACs. """
    def get_mBAC_from_formula(self, inc):
        mBAC_bonds = CBH(self.para).get_mBAC_bonds(self.para['species'][inc])
        mBAC = 0

        for k, v in mBAC_bonds.items():
            try:
                mBAC = mBAC + self.para['mBAC_parameters'][k] * v
            except:
                return None
        return mBAC


    """ To get IGCCs. """
    def get_IGCC_from_formula(self, inc):
        IGCC_bonds = CBH(self.para).get_CBH_delta_bonds(self.para['species'][inc], 3)[-1]
        IGCC = 0
        for k, v in IGCC_bonds.items():
            try:
                IGCC = IGCC + self.para['IGCC_parameters'][k] * v
            except:
                return None
        return IGCC

    

    """ To write thermodynamic files. """
    def write_thermo_dat(self):
        
        # Get all basic dat files.
        chdir(f"{self.para['work_path']}/datfiles")
        df = read_csv(f"{self.para['work_path']}/csvfiles/input_data.csv", index_col=0)
        input_data = df.to_dict(orient='index')
        df = read_csv(f"{self.para['work_path']}/csvfiles/source_data.csv", index_col=0)
        source_data = df.to_dict(orient='index')

        for method in ['B3LYP', 'MP2', 'CCSDT']:
            for inc, values in source_data.items():
                if f'{inc}.dat' not in listdir(f"{self.para['work_path']}/datfiles/{method}"):
                    try:
                        # Specify the final Heat of Formation as the result of CCSDT.
                        if method == 'CCSDT':
                            if inc in self.para['macro_mol']:
                                values[f'Hf_CCSDT/(kcal/mol)'] = values[f'Hf_CCSDT_CBH/(kcal/mol)']

                        cal_smi = self.basic.smi_to_std_format(self.para['species'][inc])[0]

                        eval(f"Thermofit('{self.para['work_path']}').output_dat_from_data('{inc}', '{cal_smi}', eval('{values[f'Hf_{method}/(kcal/mol)']}'),"
                            f" eval('{values['S_298/(cal/mol/K)']}'), eval('{values['Cp/(cal/mol/K)']}'), '{method}')")
                    except:
                        pass

        # Write thermodyamic parameters for all molecules to a dat file for Chemkin.
        with open('thermo.dat', 'w') as f:
            f.write('THERM ALL\n   300.000  1000.000  5000.000\n')
            for species, values in input_data.items():
                if f"{values['inchikey']}.dat" in listdir(f"{self.para['work_path']}/datfiles/CCSDT"):
                    with open(f"{self.para['work_path']}/datfiles/CCSDT/{values['inchikey']}.dat") as p:
                        dat = findall('.{24}((.+\n){4})', p.read())[0][0]
                        f.write('{:<24}{}'.format(species, dat))
            f.write('END\n')
        
        # Write thermodyamic parameters for macro molecules to a dat file for Chemkin.
        with open('macro_mol.dat', 'w') as f:
            f.write('THERM ALL\n   300.000  1000.000  5000.000\n')
            for species, values in input_data.items():
                if f"{values['inchikey']}.dat" in listdir(f"{self.para['work_path']}/datfiles/CCSDT"):
                    if values['inchikey'] in self.para['macro_mol']:
                        with open(f"{self.para['work_path']}/datfiles/CCSDT/{values['inchikey']}.dat") as p:
                            dat = findall('.{24}((.+\n){4})', p.read())[0][0]
                            f.write('{:<24}{}'.format(species, dat))
            f.write('END\n')
        
        # Write all thermodyamic data at 298.15 K.
        with open('all_thermo_data.txt', 'w') as f:
            f.write('T: 298.15K,  Hf: kcal/mol,  S: cal/mol/K,  Cp: cal/mol/K\n')
            f.write(f"{'species':<30}{'inchikey':40}{'smiles':60}{'Hf_B3LYP':>20}{'Hf_MP2':>20}{'Hf_CCSDT':>20}{'S':>20}{'Cp':>20}\n")
            for species, values in input_data.items():
                if all(f"{values['inchikey']}.dat" in listdir(f"{self.para['work_path']}/datfiles/{method}") for method in ['B3LYP', 'MP2', 'CCSDT']):
                    Hf_B3LYP, S, Cp = Thermofit(self.para['work_path']).get_thermo_from_data(values['inchikey'], 'B3LYP')
                    Hf_MP2, S, Cp = Thermofit(self.para['work_path']).get_thermo_from_data(values['inchikey'], 'MP2')
                    Hf_CCSDT, S, Cp = Thermofit(self.para['work_path']).get_thermo_from_data(values['inchikey'], 'CCSDT')
                    f.write(f"{species:<30}{values['inchikey']:40}{values['smiles']:60}{Hf_B3LYP:20.2f}{Hf_MP2:20.2f}{Hf_CCSDT:20.2f}{S:20.2f}{Cp:20.2f}\n")
        
        # Write all thermodyamic data for all smiles at 298.15 K.
        with open('smiles_data.txt', 'w') as f:
            f.write('T: 298.15K,  Hf: kcal/mol,  S: cal/mol/K,  Cp: cal/mol/K\n')
            f.write(f"{'inchikey':40}{'smiles':60}{'Hf_B3LYP':>20}{'Hf_MP2':>20}{'Hf_CCSDT':>20}{'S':>20}{'Cp':>20}\n")
            for inc, values in source_data.items():
                if all(f'{inc}.dat' in listdir(f"{self.para['work_path']}/datfiles/{method}") for method in ['B3LYP', 'MP2', 'CCSDT']):
                    Hf_B3LYP, S, Cp = Thermofit(self.para['work_path']).get_thermo_from_data(inc, 'B3LYP')
                    Hf_MP2, S, Cp = Thermofit(self.para['work_path']).get_thermo_from_data(inc, 'MP2')
                    Hf_CCSDT, S, Cp = Thermofit(self.para['work_path']).get_thermo_from_data(inc, 'CCSDT')
                    f.write(f"{inc:40}{values['standard_smiles']:60}{Hf_B3LYP:20.2f}{Hf_MP2:20.2f}{Hf_CCSDT:20.2f}{S:20.2f}{Cp:20.2f}\n")
 
        # Check whether all thermodynamic parameters are completed.
        if all(f'{inc}.dat' in listdir(f"{self.para['work_path']}/datfiles/CCSDT") for inc in self.para['species']):
            print(f"\n{' Thermo completed! ':*^30}\n")
            print(f"\n{'':*^30}")
            print(f"\n{' Normal termination! ':#^30}\n")
        else:
            print(f"\n{' Thermo incompleted! ':*^30}\n")
            print(f"\n{'':*^30}")
            print(f"\n{' Exit this turn! ':#^30}\n")

        return None


    """ To get all thermodynamic parameters. """
    def get_all_thermodat(self):
        print('\nProcess thermodynamic data ...\n')
        self.get_data_from_out()
        self.write_thermo_dat()

