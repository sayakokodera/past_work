# -*- coding: utf-8 -*-
######### remove units function #########

from .ultrasonic_imaging_python.definitions import units
ureg = units.ureg

def remove_units(parameter_with_unit):
    return (parameter_with_unit.to_base_units()).magnitude
