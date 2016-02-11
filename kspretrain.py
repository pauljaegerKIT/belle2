__author__ = 'pjaeger'

"""
Prepare the data for training:
Write out necessary Variables 
to accelerate workflow in a parallel processing.

"""

from basf2 import *
from stdLooseFSParticles import stdVeryLoosePi
from stdFSParticles import stdPi0
from stdV0s import stdKshorts
from modularAnalysis import *
import sys

inputMdstList(sys.argv[1])
preTrainingDir = sys.argv[3]

stdVeryLoosePi()
reconstructDecay('K_S0:ks -> pi-:all pi+:all','')
vertexKFit('K_S0:ks', -1)
applyCuts('K_S0:ks','0.477614<M<0.517614')

variables = [

    'p',
    'distance',
    'cosAngleBetweenMomentumAndVertexVector',
    'decayAngle(0)',
    'decayAngle(1)',
    'daughter(0,distance)',
    'daughter(1,distance)',
]

teacher = register_module('TMVATeacher')
teacher.param('workingDirectory', preTrainingDir)
teacher.param('prefix', sys.argv[2])
teacher.param('variables', variables)
teacher.param('spectators', ['isMySignal'])
teacher.param('listNames', 'K_S0:ks')
analysis_main.add_module(teacher)
process(analysis_main)

print(statistics)
