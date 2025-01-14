#include <host_support.h>
#include <stdio.h>
#include <malloc.h>
#include <sys/time.h>
#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>
#include "../../vitis-power/vitis-power/vitis-power.hpp"

// Datatype to use, must match the kernel datatype
#define DATA_TYPE double
// Memory page size, ensures that memory is aligned to the boundary for performance
#define PAGESIZE 4096

static void init_device(char*, int);
static void execute_on_device(cl::Event&,cl::Event&,cl::Event&);
static void init_problem(int);
static float getTimeOfComponent(cl::Event&);

DATA_TYPE *input_data, *result_data; // Input and result data

cl::CommandQueue * command_queue;
cl::Context * context;
cl::Program * program;
cl::Kernel * sum_kernel;
cl::Buffer *buffer_input, *buffer_result; // Buffers to transfer to and from the device

int main(int argc, char * argv[]) {    
  cl::Event copyOnEvent, kernelExecutionEvent, copyOffEvent;
  double cardPowerAvgInWatt;

  if (argc != 3) {
    printf("You must supply two command line arguments, the bitstream file and number of data elements\n");
    return EXIT_FAILURE;
  }
  int data_size=atoi(argv[2]);

  init_problem(data_size);
  init_device(argv[1], data_size);


  bool stop_measurement = false;
  #pragma omp parallel shared(stop_measurement, cardPowerAvgInWatt, copyOnEvent, kernelExecutionEvent, copyOffEvent) num_threads(2)
  {
      int tid = omp_get_thread_num();
      if(tid == 0) {
        execute_on_device(copyOnEvent, kernelExecutionEvent, copyOffEvent);
      } else {
          bool profile_power=true;
          cardPowerAvgInWatt = vitis_power::U280::measureFpgaPower(stop_measurement, profile_power);
      }
      stop_measurement = true;
  }
  
  float kernelTime=getTimeOfComponent(kernelExecutionEvent);
  float copyOnTime=getTimeOfComponent(copyOnEvent);
  float copyOffTime=getTimeOfComponent(copyOffEvent);
  
  printf("Total runtime: %f ms, (%f ms xfer on, %f ms execute, %f ms xfer off) for %d elements; Avg card power: %f W; kernelTimeEnergy: %f J\n", copyOnTime+kernelTime+copyOffTime, copyOnTime, kernelTime, copyOffTime, data_size, cardPowerAvgInWatt, cardPowerAvgInWatt*(kernelTime/1000));    
 
  delete buffer_input;
  delete buffer_result;
  delete sum_kernel;
  delete command_queue;
  delete context;
  delete program;
  
  return EXIT_SUCCESS;
}

/**
* Retrieves the time in milliseconds of the OpenCL event execution
*/
static float getTimeOfComponent(cl::Event & event) {
  cl_ulong tstart, tstop;

  event.getProfilingInfo(CL_PROFILING_COMMAND_START, &tstart);
  event.getProfilingInfo(CL_PROFILING_COMMAND_END, &tstop);
  return (tstop-tstart)/1.E6;
}

/**
* Performs execution on the device by transfering input data, running the kernel, and copying result data back
* We use OpenCL events here to set the dependencies properly
*/
static void execute_on_device(cl::Event & copyOnEvent, cl::Event & kernelExecutionEvent, cl::Event & copyOffEvent) {
  cl_int err;

  // Queue migration of memory objects from host to device (last argument 0 means from host to device)
  OCL_CHECK(err, err = command_queue->enqueueMigrateMemObjects({*buffer_input}, 0, nullptr, &copyOnEvent));	

  // Queue kernel execution
  std::vector<cl::Event> kernel_wait_events;
  kernel_wait_events.push_back(copyOnEvent);
  OCL_CHECK(err, err = command_queue->enqueueTask(*sum_kernel, &kernel_wait_events, &kernelExecutionEvent));

  // Queue copy result data back from kernel
  std::vector<cl::Event> data_transfer_wait_events;
  data_transfer_wait_events.push_back(kernelExecutionEvent);
  OCL_CHECK(err, err = command_queue->enqueueMigrateMemObjects({*buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST, &data_transfer_wait_events, &copyOffEvent));

  // Wait for queue to complete
  OCL_CHECK(err, err = command_queue->finish());
}

/**
* Initiates the FPGA device and sets up the OpenCL context
*/
static void init_device(char * binary_filename, int data_size) {
  cl_int err;

  std::vector<cl::Device> devices;
  std::tie(program, context, devices)=initialiseDevice("Xilinx", "u280", binary_filename);

  // Create the command queue (and enable profiling so we can get performance data back)
  OCL_CHECK(err, command_queue=new cl::CommandQueue(*context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err));
  // Create a handle to the sum kernel
  OCL_CHECK(err, sum_kernel=new cl::Kernel(*program, "sum_kernel", &err));
  // Allocate global memory OpenCL buffers that will be copied on and off
  OCL_CHECK(err, buffer_input=new cl::Buffer(*context, CL_MEM_USE_HOST_PTR  | CL_MEM_READ_ONLY, data_size * sizeof(DATA_TYPE), input_data, &err));
  OCL_CHECK(err, buffer_result=new cl::Buffer(*context, CL_MEM_USE_HOST_PTR  | CL_MEM_WRITE_ONLY, data_size * sizeof(DATA_TYPE), result_data, &err));

  // Set kernel arguments
  OCL_CHECK(err, err = sum_kernel->setArg(0, *buffer_input));
  OCL_CHECK(err, err = sum_kernel->setArg(1, *buffer_result));  
  OCL_CHECK(err, err = sum_kernel->setArg(2, (DATA_TYPE) 50)); // Add 50 to each element
  OCL_CHECK(err, err = sum_kernel->setArg(3, data_size));
}

/**
* Initialises the underlying input and result data (and ensures these are aligned to page boundaries for performance) along
* with setting the initial input data
*/
static void init_problem(int data_size) {
  input_data=(DATA_TYPE*) memalign(PAGESIZE, sizeof(DATA_TYPE) * data_size);
  result_data=(DATA_TYPE*) memalign(PAGESIZE, sizeof(DATA_TYPE) * data_size);
  for (int i=0;i<data_size;i++) {
    input_data[i]=(DATA_TYPE) i;
  }
}

