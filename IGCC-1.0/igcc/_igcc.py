# -*- coding: utf-8 -*-
from os.path import abspath, dirname

from igcc._submit import Submit
from igcc._thermo import Thermo
from igcc._parameters import Parameters



class IGCC(object):
    
    """ To initiate parameters for TDOC. """
    def __init__(self, input_file, para_file):
        super(IGCC, self).__init__()
        self.input_file = input_file
        self.para_file = para_file
        self.work_path = dirname(abspath(self.input_file)).replace('\\\\', '/').replace('\\','/')


 
    """ To execute programm. """
    def execute(self):
        parameters = Parameters(self.input_file, self.para_file, self.work_path)
        self.para = parameters.get_all_parameters()
        submit = Submit(self.para)
        submit.get_submitted_out()
        thermo = Thermo(self.para)
        thermo.get_all_thermodat()
      