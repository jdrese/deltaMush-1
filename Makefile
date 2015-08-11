ifeq ($(USER), giordi)
include buildconfig_home
endif


.SUFFIXES: .cpp .o .cu .h
TARGET= deltaMush.so
TOP=/usr/autodesk/maya2016/devkit/plug-ins
SRCDIR=.


#CUDA_LIB = -lcudart -lcudadevrt
#CUDA_PATH = "/usr/local/cuda-6.5"
#CUDA_LIB_PATH = -L /usr/local/cuda/lib64
#NVCC = $(CUDA_PATH)/bin/nvcc
#CUDA_FLAGS =  -arch=sm_30 --compiler-options '-fPIC'

all : deltaMush.o pluginMain.o 
	$(LD)    $? -o $(TARGET) $(LFLAGS) $(LIBS) -lOpenMaya -lOpenMayaAnim -lFoundation 
%.o: %.cpp
	$(CXX) $(C++FLAGS) -ftemplate-depth=50  $(INCLUDES) -c $< -o $@ -std=c++11
%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@
clean:
	rm -f *.o
	rm -f *.so


#g++ -DBits64_ -m64 -DUNIX -D_BOOL -DLINUX -DFUNCPROTO -D_GNU_SOURCE -DLINUX_64 -fPIC -fno-strict-aliasing -DREQUIRE_IOSTREAM -O3 -Wall -Wno-multichar -Wno-comment -Wno-sign-compare -funsigned-char -pthread  -Wno-deprecated -Wno-reorder -ftemplate-depth-25 -fno-gnu-keywords  -I . -I/usr/autodesk/maya2016/include -I/usr/X11R6/include -c MG_chainOnPath.cpp


#g++     MG_chainOnPath.o pluginMain.o -o MG_chainOnPath.so -DBits64_ -m64 -DUNIX -D_BOOL -DLINUX -DFUNCPROTO -D_GNU_SOURCE -DLINUX_64 -fPIC -fno-strict-aliasing -DREQUIRE_IOSTREAM -O3 -Wall -Wno-multichar -Wno-comment -Wno-sign-compare -funsigned-char -pthread  -Wno-deprecated -Wno-reorder -ftemplate-depth-25 -fno-gnu-keywords -Wl,-Bsymbolic -shared -Wl,--version-script=/usr/autodesk/maya2016/devkit/plug-ins/linux_plugin.map -L/usr/autodesk/maya2016/lib -lOpenMaya -lFoundation

