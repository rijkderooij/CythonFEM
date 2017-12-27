# -*- coding: utf-8 -*-
import pyximport; 
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import time
import os, sys

# Add paths
myPath = os.getcwd()
sys.path.insert(0, myPath+'/../pyxd')
sys.path.insert(0, myPath+'/InputFiles')
pyximport.install(setup_args={"include_dirs":np.get_include()})

from DataContainer import *
from ModelContainer import *
from SOL2 import *
from EllipseMF_Q2 import *
import ElementHelper as eh
import OutputHelper as oh


#  ================================================  #
#  -----------  M A I N  P R O G R A M  -----------  #
#  ------------------------------------------------  #

# ModelContainer
mc = EllipseMF_Q2()
mc.BuildModel(None);
print mc

# DataContainer
dc = DataContainer();
dc.InitializeData(mc);

# Plot
# Initialize data
for el in mc.elements:
	el.InitializeData(dc)

# Share initialized data
dc.ShareData()
oh.ParaviewOutput(mc,dc) # Write paraview output


# Solver
sc = SOL2(mc,dc,True,2)
sc.maxIter = 10;
sc.maxIterInc = 6;
sc.maxStep = 2000;
sc.tEnd = 5.;
sc.dt0  = 0.25;
sc.dtMin = sc.dt0/10000.;
sc.dtMax = sc.tEnd/10.;

startTime = time.time()
sc.Solve()
endTime = time.time()
oh.WriteToLog(mc,"Total time elapsed for solving: "+str(endTime-startTime))

