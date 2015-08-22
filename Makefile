
#configuring home enviroment
ifeq ($(USER), giordi)
include buildconfig_home
LFAGS += -L /home/giordi/WORK_IN_PROGRESS/C/libs
TOP= /usr/autodesk/maya2016/devkit/plug-ins
INCLUDES += -isystem /home/giordi/WORK_IN_PROGRESS/C/libs/eigen

CUDA_LIB = -lcudart -lcudadevrt
CUDA_PATH = "/usr/local/cuda-6.5"
CUDA_LIB_PATH = -L /usr/local/cuda/lib64
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_FLAGS =  -arch=sm_30 --compiler-options '-fPIC'

#configuring other enviroment
else ifeq ($(USER),mog)
include buildconfig
TOP= $(MAYA_LOCATION)/devkit/plug-ins
#override gcc
CXX= gcc481
endif

#defining source directory needed by build config
SRCDIR=.

BUILD := release
ifeq ($(BUILD),debug)
C++FLAGS += -g
endif

#adding custom flags
C++FLAGS += -funroll-loops -msse4 -DMaya$(mayaVersion)

#extra flags for maya 2016
ifeq ($(mayaVersion),2016)
C++FLAGS += -ftemplate-depth=50 -std=c++11 
endif

.SUFFIXES: .cpp .o .cu .h

#compiling object and targets
TARGET= deltaMush.so
OBJS = deltaMush.o pluginMain.o  deltaMushOpencl.o

all : $(OBJS) 
	$(LD)    $? -o $(TARGET) $(LFLAGS) $(LIBS) -lOpenMaya -lOpenMayaAnim -lFoundation -ltbb
%.o: %.cpp
	$(CXX) $(C++FLAGS) $(INCLUDES) -c $< -o $@ 
%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@
clean:
	rm -f *.o
	rm -f *.so


#g++ -DBits64_ -m64 -DUNIX -D_BOOL -DLINUX -DFUNCPROTO -D_GNU_SOURCE -DLINUX_64 -fPIC -fno-strict-aliasing -DREQUIRE_IOSTREAM -O3 -Wall -Wno-multichar -Wno-comment -Wno-sign-compare -funsigned-char -pthread  -Wno-deprecated -Wno-reorder -ftemplate-depth-25 -fno-gnu-keywords  -I . -I/usr/autodesk/maya2016/include -I/usr/X11R6/include -c MG_chainOnPath.cpp


#g++     MG_chainOnPath.o pluginMain.o -o MG_chainOnPath.so -DBits64_ -m64 -DUNIX -D_BOOL -DLINUX -DFUNCPROTO -D_GNU_SOURCE -DLINUX_64 -fPIC -fno-strict-aliasing -DREQUIRE_IOSTREAM -O3 -Wall -Wno-multichar -Wno-comment -Wno-sign-compare -funsigned-char -pthread  -Wno-deprecated -Wno-reorder -ftemplate-depth-25 -fno-gnu-keywords -Wl,-Bsymbolic -shared -Wl,--version-script=/usr/autodesk/maya2016/devkit/plug-ins/linux_plugin.map -L/usr/autodesk/maya2016/lib -lOpenMaya -lFoundation

