from ModelContainer cimport *
cdef class EllipseMF_Q2(ModelContainer):
	cpdef BuildModel(self,object p)
	cpdef GetElemDir(self, double x, double y, str option)
	cpdef ReadAbaqusFile(self,str filename)
