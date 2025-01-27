.RECIPEPREFIX +=
#######################################################################################
.PHONY: help
help:
  @echo "Makefile Usage:"
  @echo "  make device TARGET=<sw_emu/hw_emu/hw>"
  @echo "      Command to generate the design for specified Target and Device."
  @echo ""
  @echo "  make host "
  @echo "      Command to generate host."
  @echo ""
  @echo "  make clean "
  @echo "      Command to remove the generated files."
  @echo ""
#######################################################################################
TARGET := sw_emu
PLATFORM := xilinx_u280_xdma_201920_3
#PLATFORM := xilinx_u250_gen3x16_xdma_3_1_202020_1
HOST_EXE := host
XO := sum_kernel.$(TARGET).xo
XCLBIN := sum_kernel.$(TARGET).xclbin

KERNEL_NAME := sum_kernel

DEVICE_SRCDIR:=src/device
TEMP_DIR := ./_x.$(TARGET)

SOURCES  := $(wildcard $(DEVICE_SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(DEVICE_SRCDIR)/*.h)
OBJECTS  := $(SOURCES:$(DEVICE_SRCDIR)/%.cpp=%.xo)

# Host building global settings
CXXFLAGS := -I$(XILINX_XRT)/include/ -I$(XILINX_VIVADO)/include/ -Wall -O3 -std=c++11 -L$(XILINX_XRT)/lib/ -lhostsupport -lpthread -lrt -lstdc++ -fopenmp
CXXFLAGS2 := -lxilinxopencl

# Kernel compiler & linker global settings
KRNL_COMPILE_OPTS := -t $(TARGET) --config ../design.cfg --save-temps -O3
KRNL_LINK_OPTS := -t $(TARGET) --config ../link.cfg -O3

build:  $(XO) $(XCLBIN) $(HOST_EXE) emconfig
  
.PHONY: device
.ONESHELL:
device: mkrefdir $(OBJECTS) $(XCLBIN)
  cp reference_files_$(TARGET)/$(XCLBIN) bin/.

mkrefdir:
  rm -Rf reference_files_$(TARGET)
  mkdir -p reference_files_$(TARGET)
  mkdir -p bin

# Building kernel
.PHONY: $(OBJECTS)
$(OBJECTS): %.xo : $(DEVICE_SRCDIR)/%.cpp
  cd reference_files_$(TARGET) ; v++ $(KRNL_COMPILE_OPTS) -c -k sum_kernel -I'../include' -I'../$(<D)' -o'$@' ../$<

.PHONY: $(XCLBIN)
$(XCLBIN): $(OBJECTS)
  cd reference_files_$(TARGET) ; v++ $(KRNL_LINK_OPTS) -l -o'$@' $(+)

.PHONY: $(HOST_EXE)
# Building Host
$(HOST_EXE): src/host/host.cpp
  mkdir -p bin
  g++ $(CXXFLAGS) -o bin/host '$<' $(CXXFLAGS2)

.PHONY: emconfig
emconfig:
  mkdir -p bin
  cd bin
  emconfigutil --platform $(PLATFORM)

# Cleaning stuff
.PHONY: clean
clean:
  rm -f $(HOST_EXE) *mmult.$(TARGET).$(PLATFORM).* *.log *.json *.xo
