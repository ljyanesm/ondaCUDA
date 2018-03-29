# Propagador de Ondas 3D Secuencial
######## Babilonia #####################################
#CC1           = g++
#CFLAGS        = -march=k8 -O3
#CFLAGS2        = -DMPICH_IGNORE_CXX_SEEK
######## Ultra 40 ######################################
#CC1           = g++
#CFLAGS        = -m64 -march=opteron -O3
#CFLAGS        = -m64 -march=opteron -O3 -Wall -Wextra -W -Wconversion
#CFLAGS2        = -DMPICH_IGNORE_CXX_SEEK
######## GENERICA #############################################################
#CC1           = icpc 
#NVCC1	      = nvcc
#CFLAGS        = -O3
#CFLAGS2       = -DMPICH_IGNORE_CXX_SEEK
######### Fast #####################################
CC1           = g++
CFLAGS        = -fast -O3
CFLAGS2        = -DMPICH_IGNORE_CXX_SEEK
###############################################################################

LIBS          = 
TARGET1       = cSwapBytes
TARGET2       = seis3D_2orden_multiGPU
TARGET3				= ondaGPU
DEL_FILE      = rm

####### Build rules
all: $(TARGET3) $(TARGET1) $(TARGET2)

$(TARGET3):
	$(NVCC1) -O3 -c $(TARGET3).cu --ptxas-options=-v
$(TARGET1):
	$(CC1) $(CFLAGS) -c $(TARGET1).cpp $(LIBS) 

$(TARGET2): $(TARGET1) $(TARGET3)
	$(CC1) $(CFLAGS) -o $(TARGET2) $(TARGET2).cpp $(TARGET1).o $(TARGET3).o $(LIBS) -lm -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64/ -lcudart

clean:
	$(DEL_FILE) -f $(TARGET1) $(TARGET2) $(TARGET3)
	$(DEL_FILE) -f *~ core *.core *.o
