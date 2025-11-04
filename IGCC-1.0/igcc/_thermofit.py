# -*- coding: utf-8 -*-
import numpy as np
from rdkit import Chem
from re import findall
from scipy import optimize
from collections import Counter
from ase.formula import Formula



class Thermofit(object):

    """ To initiate parameters for Thermo. """
    def __init__(self, work_path):
        self.work_path = work_path



    """ NASA polynomial for Hf. """
    def func_Hf(self, T, a1, a2, a3, a4, a5, a6):
        return (a1 * T + a2/2 * T ** 2 + a3 / 3 * T ** 3 + a4 / 4 * T ** 4 + a5/5 * T ** 5 + a6) * 1.987



    """ NASA polynomial for S. """
    def func_S(self, T, a1, a2, a3, a4, a5, a7):
        return (a1 * np.log(T) + a2 * T + a3 / 2 * T ** 2 + a4 / 3 * T ** 3 + a5/4 * T ** 4 + a7) * 1.987



    """ NASA polynomial for Cp. """
    def func_Cp(self, T, a1, a2, a3, a4, a5):
        return (a1 + a2 * T + a3 * T ** 2 + a4 * T ** 3 + a5 * T ** 4) * 1.987



    """ NASA polynomial for Cp by piecewise fitting. """
    def piecewise_fitting_Cp(self, T, p0, p1, p2, p3, p4, p7, p8, p9, p10, p11):
        return np.piecewise(T, [T >= 1000, T < 1000], [lambda T:(p0 + p1 * T + p2 * T ** 2 + p3 * T ** 3 + p4 * T ** 4) * 1.987, lambda T:(p7 + p8 * T + p9 * T ** 2 + p10 * T ** 3 + p11 * T ** 4) *1.987])



    """ To get thermal data from a dat file. """
    def get_thermo_from_data(self, inc, method):

        # Get thermodyamic data from NASA parameters.
        with open(f'{self.work_path}/datfiles/{method}/{inc}.dat') as f:
            content = f.read()
        res = list(map(float, content[45:75].split()))
        para = list(map(float, findall('.{15}', findall('\n([\s\S]+)\s{10}', content)[0])))
        Hf, S, Cp = self.func_Hf(298.15, *para[7: 13]) / 1000, self.func_S(298.15, *(para[7: 12] + para[13:])), self.func_Cp(298.15, *para[7: 12])
        return round(Hf, 2), round(S, 2), round(Cp, 2)



    def output_dat_from_data(self, inc, newsmi, Hf, S, Cp, method):
        
        # Get 14 NASA parameters.
        T_list = sorted(Cp)
        Cp_list = [Cp[T] for T in T_list]
        p0, p1, p2, p3, p4, p7, p8, p9, p10, p11 = optimize.curve_fit(self.piecewise_fitting_Cp, T_list, Cp_list)[0]
        para = [p0, p1, p2, p3, p4, 0, 0, p7, p8, p9, p10, p11, 0, 0]
        para[12] = Hf *1000 / 1.987 - self.func_Hf(298.15, *para[7:13]) / 1.987
        para[5] = self.func_Hf(1000, *para[7:13]) / 1.987 - self.func_Hf(1000, *para[:6]) / 1.987
        para[13] = S / 1.987 - self.func_S(298.15, *para[7:12], para[13]) / 1.987
        para[6] = self.func_S(1000, *para[7:12], para[13]) / 1.987 - self.func_S(1000, *para[:5], para[6]) / 1.987
        
        # Output thermodynamic parameters for Chemkin.
        atoms = Counter(x.GetSymbol() for x in Chem.AddHs(Chem.MolFromSmiles(newsmi)).GetAtoms())
        atomlist = ''.join('{:<2}{:>3}'.format(x, atoms[x]) for x in atoms)
        species = ''.join(x + str(atoms[x]) if atoms[x] != 1 else x for x in atoms)

        with open(f'{self.work_path}/datfiles/{method}/{inc}.dat', 'w') as f:
            f.write(f"{species:<24}{atomlist:<20}G{T_list[0]:>10.3f}{T_list[-1]:>10.3f}{1000.00:>8.2f}{1:>7}\n")
            f.write(f"{para[0]:>15.8E}{para[1]:>15.8E}{para[2]:>15.8E}{para[3]:>15.8E}{para[4]:>15.8E}{2:>5}\n")
            f.write(f"{para[5]:>15.8E}{para[6]:>15.8E}{para[7]:>15.8E}{para[8]:>15.8E}{para[9]:>15.8E}{3:>5}\n")
            f.write(f"{para[10]:>15.8E}{para[11]:>15.8E}{para[12]:>15.8E}{para[13]:>15.8E}{4:>20}\n")
