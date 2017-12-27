from ModelContainer cimport *
from Amplitude cimport *
from MAT_NEOH cimport *
from MAT_VISC cimport *
from PBAR cimport *
from PSOLID71 cimport *
from Node cimport *
from CBAR7 cimport *
from CQUAD7 cimport *
from SPC cimport *
from LOAD cimport *

from math import *
import numpy as np
cimport numpy as np

cdef class EllipseMF_Q2(ModelContainer):
	def __init__(self):
		super().__init__()
		print "Initializing Rectangle_Q2"

	cpdef BuildModel(self, object p):
		self.numDofPerNode = 3
		self.numDispDofPerNode = 2
		
		# Number of elements in circumferential direction (15,30,45,60,75,90,105,120)
		cdef int nelc = 60

		# Read mesh
		cdef np.ndarray[double,ndim=2] nod 
		cdef np.ndarray[np.int_t,ndim=2] elem
		cdef np.ndarray[np.int_t,ndim=1] cortex_elem, subcor_elem, core_elem
		cdef str filename = 'nc_'+str(nelc)+'.inp'
		nod, elem, cortex_elem, subcor_elem, core_elem = self.ReadAbaqusFile(filename)

		# Perturbation
		cdef double sizeY_cortex = 2.0
		cdef double pert = 0.005            # pertubation of center nodes in cortex (normalized wrt cortical thickness)
		cdef double pert_width = 0.05       # width of perturbed area (normalized wrt width of plate)

		# Compute sizes and numbers
		cdef double sizeX = np.max(nod[:,1])
		cdef double sizeY = np.max(nod[:,2])
		cdef double sizeY_subcortex = sizeY-sizeY_cortex
		cdef int nel_cortex = len(cortex_elem)
		cdef int nel_subcortex = len(subcor_elem)
		cdef int nel_core = len(core_elem)
		cdef int nel = nel_cortex+nel_subcortex+nel_core
		cdef int nnode = len(nod)

		# Element direction
		cdef np.ndarray[double, ndim=1] elemDir_cortex
		cdef str elemDir_subcortex
		elemDir_cortex = np.array([0.,1.])
		elemDir_subcortex = 'vertical';       # random/horizontal/vertical/radial
		np.random.seed(0);

		# Material and growth parameters
		cdef double D  = 70. 				# Diffusivity
		cdef double Grho = 2.0      	    # Limit density
		cdef double kth1 = 50.         		# dth1/drho
		cdef double kth2 = 0.         		# dth2/drho
		cdef double eta = .01;			# Surrounding viscosity

		# Materials
		cdef dict matProp = {'E':10.0,'nu':0.4,'D':D, 'eta':eta}
		self.materials = [];
		self.materials.append(MAT_NEOH(matProp))
		self.materials[0].localID  = 12;
		matProp['E']/=5
		self.materials.append(MAT_NEOH(matProp))
		self.materials[1].localID  = 23;

		self.materials.append(MAT_VISC(matProp))
		self.materials[2].localID  = 34;

		# Create amplitude
		self.amplitudes.append(Amplitude(np.array([0.,2000.]),np.array([0.,0.])));

		# Property
		cdef dict propProp = {'area':1.0, 'kth1':kth1,'kth2':kth2, 'Grho':0.}
		self.properties = [];
		self.properties.append(PSOLID71(self.materials[0],propProp))	# Cortex
		self.properties[0].localID = 111;

		propProp['kth1'] = 0.; propProp['kth2']=0.;
		propProp['Grho'] = 0.;
		self.properties.append(PSOLID71(self.materials[1],propProp))	# Subcortex
		self.properties[1].localID = 222;
		
		propProp['Grho'] = Grho;
		self.properties.append(PSOLID71(self.materials[1],propProp)) 	# Core
		self.properties[2].localID = 333;

		self.properties.append(PBAR(self.materials[2],propProp)) 	# Core
		self.properties[3].localID = 444;

		for p in self.properties:
			print p

		# Nodes
		cdef int nid;
		cdef double rMin = 1.e9;
		cdef list nidVisc = [];
		cdef list nidViscFix = [];
		cdef double xi, yi
		for nid in range(len(nod)):
			xi = nod[nid,1]
			yi = nod[nid,2]
			thi = np.arctan2(yi,xi);

			if (xi/sizeX)**2+(yi/sizeY)**2>1.-0.0001:
				nidVisc.append(nid);

			if(np.abs(thi-pi/2.)<=pert_width and yi>=sizeY_subcortex-1.e-6):
				yi += pert*sizeY_cortex

			self.nodes.append(Node([xi,yi]));
			self.nodes[nid].localID = nid;
			self.nodes[nid].dofID = range(nid*self.numDofPerNode,(nid+1)*self.numDofPerNode);

		# Create two nodes to attach all viscous dampers
		nid+=1;
		self.nodes.append(Node([0.,0.]));
		self.nodes[nid].localID = nid;
		self.nodes[nid].dofID = range(nid*self.numDofPerNode,(nid+1)*self.numDofPerNode);
		nidViscFix.append(nid);
		nid+=1;
		self.nodes.append(Node([-sizeX,sizeY]));
		self.nodes[nid].localID = nid;
		self.nodes[nid].dofID = range(nid*self.numDofPerNode,(nid+1)*self.numDofPerNode);
		nidViscFix.append(nid);
		nid+=1;
		self.nodes.append(Node([sizeX,sizeY]));
		self.nodes[nid].localID = nid;
		self.nodes[nid].dofID = range(nid*self.numDofPerNode,(nid+1)*self.numDofPerNode);
		nidViscFix.append(nid);

		# Elements
		cdef int count, eid
		cdef np.ndarray[np.int_t, ndim=1] elnod
		cdef np.ndarray[double, ndim=1] locAvg, elemDir
		cdef np.ndarray[double, ndim=2] locN
		cdef double xA, yA
		count = 0

		# core
		for eid in range(nel_core):
			elnod = elem[core_elem[eid],1:5]

			# Compute center of element
			locN = np.array([self.nodes[elnod[0]].loc, self.nodes[elnod[1]].loc,\
			                 self.nodes[elnod[2]].loc, self.nodes[elnod[3]].loc])

			locAvg = np.array([np.mean(locN[:,i]) for i in range(len(locN[0]))])
			xA = locAvg[0]
			yA = locAvg[1]

			# Element direction
			elemDir = self.GetElemDir(xA,yA,elemDir_subcortex)

			# Create element
			self.elements.append(CQUAD7([self.nodes[int(ii)] for ii in elnod],elemDir,self.properties[2]));
			self.elements[count].localID = count;

			count+=1


		# subcortex
		for eid in range(nel_subcortex):
			elnod = elem[subcor_elem[eid],1:5]

			# Compute center of element
			locN = np.array([self.nodes[elnod[0]].loc, self.nodes[elnod[1]].loc,\
			                 self.nodes[elnod[2]].loc, self.nodes[elnod[3]].loc])

			locAvg = np.array([np.mean(locN[:,i]) for i in range(len(locN[0]))])
			xA = locAvg[0]
			yA = locAvg[1]

			# Element direction
			elemDir = self.GetElemDir(xA,yA,elemDir_subcortex)

			# Create element
			self.elements.append(CQUAD7([self.nodes[int(ii)] for ii in elnod],elemDir,self.properties[1]));
			self.elements[count].localID = count;

			count+=1

		# cortex
		for eid in range(nel_cortex):
			elnod = elem[cortex_elem[eid],1:5]

			# Compute elemDir tangential to cortex
			locAvg = np.array([0.,0.]);
			for ii in elnod:
				locAvg+=np.array(self.nodes[int(ii)].loc)
			th = np.arctan2(locAvg[1]*sizeX,locAvg[0]*sizeY);
			dydx = -sizeY/sizeX/np.tan(th);

			elemDir = np.array([-dydx,1.]); elemDir /= np.linalg.norm(elemDir);

			# Create element
			self.elements.append(CQUAD7([self.nodes[int(ii)] for ii in elnod],elemDir,self.properties[0]));
			self.elements[count].localID = count;

			count+=1

		# viscous elements
		for nidf in nidViscFix:
			for nid in nidVisc:
				self.elements.append(CBAR7([self.nodes[nid], self.nodes[nidf]],self.properties[3]));
				self.elements[count].localID = count;
				count+=1


		# SPC
		for n in self.nodes:
			# clamp bottom nodes
			if(n.y < 1e-6):
				self.spc.append(SPC(n,range(self.numDispDofPerNode),0.,self.amplitudes[0]))
		# clamp attachment nodes of bar elements and fix density
		for nidf in nidViscFix:
			self.spc.append(SPC(self.nodes[nidf],range(self.numDofPerNode),0.,self.amplitudes[0]))
			
		
		# Loads
		# - 

		# length of all matrices
		self.SetListLengths()

	cpdef GetElemDir(self, double x, double y, str option):
		cdef np.ndarray[double,ndim=1] elemDir
		
		if(option=='random'):
			th = 2*pi*np.random.rand()
			elemDir = np.array([cos(th), sin(th)])
		elif(option=='horizontal'):
			elemDir = np.array([1.,0.])
		elif(option=='vertical'):
			elemDir = np.array([0.,1.])
		elif(option=='diagonal'):
			elemDir = np.array([1.,1.])/sqrt(2.)
		elif(option=='radial'):
			elemDir = np.array([x,y])
			elemDir/= np.norm(elemDir)
		else:
			print "ERROR:  elemDir option ", option, " is not supported"
		
		return elemDir

	cpdef ReadAbaqusFile(self,str filename):
		# Open file
		fd = open('InputFiles/'+filename,'r')

		# Maximum number of lines
		maxLine = 1000000;

		##########
		### NODES
		##########
		fd.seek(0)
		iterr = 0
		found = 0
		line = fd.readline()
		while line!="" and iterr<maxLine:
			iterr+=1
			if '*Node' in line:
				found = 1
				break
			line = fd.readline()

		#Read lines
		iterr = 0
		line = fd.readline()
		nodes = []
		while line!="" and iterr<maxLine:
			data = line.strip().split(',')
			try:
				nodes.append([float(data[i]) for i in range(len(data))]);
			except:
				break


			line = fd.readline()
			iterr+=1

		# Write as array
		nodes=np.array([np.array(nodi) for nodi in nodes])	

		##########
		# ELEMENTS
		##########
		fd.seek(0)
		iterr = 0
		found = 0
		line = fd.readline()
		while line!="" and iterr<maxLine:
			iterr+=1
			if '*Element' in line:
				found = 1
				break
			line = fd.readline()

		#Read lines
		iterr = 0
		line = fd.readline()
		elements = []
		while line!="" and iterr<maxLine:
			data = line.strip().split(',')
			try:
				elements.append([int(data[i]) for i in range(len(data))]);
			except:
				break

			line = fd.readline()
			iterr+=1

		# Write as array
		elements=np.array([np.array(eli) for eli in elements])

		##########
		# CORTEX ELEMENTS
		##########
		fd.seek(0)
		iterr = 0
		line = fd.readline()
		while line!="" and iterr<maxLine:
			iterr+=1
			if '*Elset, elset=ELLIPSE_FACE-CORTEX, generate' in line:
				line = fd.readline()
				try:
					data = line.strip().split(',')
					cortex_elements = range(int(data[0]),int(data[1])+1,int(data[2]))
				except:
					break
				break
			line = fd.readline()
			iterr+=1

		# Write as array
		cortex_elements=np.array(cortex_elements)

		##########
		# SUBCORTEX ELEMENTS
		##########
		fd.seek(0)
		iterr = 0
		line = fd.readline()
		while line!="" and iterr<maxLine:
			iterr+=1
			if '*Elset, elset=ELLIPSE_FACE-SUBCORTEX, generate' in line:
				line = fd.readline()
				try:
					data = line.strip().split(',')
					subcortex_elements = range(int(data[0]),int(data[1])+1,int(data[2]))
				except:
					break
				break
			line = fd.readline()
			iterr+=1

		# Write as array
		subcortex_elements=np.array(subcortex_elements)

		##########
		# CORE ELEMENTS
		##########
		fd.seek(0)
		iterr = 0
		line = fd.readline()
		while line!="" and iterr<maxLine:
			iterr+=1
			if '*Elset, elset=ELLIPSE_FACE-CORE, generate' in line:
				line = fd.readline()
				try:
					data = line.strip().split(',')
					core_elements = range(int(data[0]),int(data[1])+1,int(data[2]))
				except:
					break
				break
			line = fd.readline()
			iterr+=1

		# Write as array
		core_elements=np.array(core_elements)

		##########
		# FINAL CONSIDERATIONS
		##########
		fd.close()

		# Account for the fact that abaqus starts at 1, python at 0
		for i in range(len(nodes)):
			nodes[i,0]-=1 
		
		elements-=1
		cortex_elements-=1
		subcortex_elements-=1
		core_elements-=1

		return (nodes, elements, cortex_elements, subcortex_elements,core_elements)