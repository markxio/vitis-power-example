#include "hls_stream.h"

#define PRAGMA_SUB(x) _Pragma (#x)
#define PRAGMA(x) PRAGMA_SUB(x)

// Defines the data type and external memory access width (ideally to make 512 bit width accesses)
#define DATA_TYPE double
#define EXTERNAL_DATA_WIDTH 8

// The data structure used to pack external memory accesses
struct packed_data {
  DATA_TYPE data[EXTERNAL_DATA_WIDTH];
};

static void retrieve_data(struct packed_data*, hls::stream<DATA_TYPE>&, int);
static void do_compute(hls::stream<DATA_TYPE>&, hls::stream<DATA_TYPE>&, DATA_TYPE, int);
static void write_data(hls::stream<DATA_TYPE>&, struct packed_data*, int);

/**
 * This is the kernel entry point, defining the protocol used for the different kernel inputs and sets up
 * a dataflow arrangement of the three functions, streaming data continually from one to the next
 */
extern "C" {
void sum_kernel(struct packed_data * input, struct packed_data  * result, DATA_TYPE add_val, int num_its) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=result bundle=control
#pragma HLS INTERFACE s_axilite port=add_val bundle=control
#pragma HLS INTERFACE s_axilite port=num_its bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  hls::stream<DATA_TYPE> input_stream, result_stream;
#pragma HLS STREAM variable=input_stream depth=8
#pragma HLS STREAM variable=result_stream depth=8

#pragma HLS DATAFLOW
  retrieve_data(input, input_stream, num_its);
  do_compute(input_stream, result_stream, add_val, num_its);
  write_data(result_stream, result, num_its);
}
}

/**
 * Retrieves external data and streams to the next dataflow stage, for efficiency will load in chunks
 * which is faster than individual elements
 */
static void retrieve_data(struct packed_data * input, hls::stream<DATA_TYPE> & stream, int num_its) {
  const unsigned int base_its=num_its / EXTERNAL_DATA_WIDTH;
  main_load_data_loop:
  for (unsigned int i=0;i<base_its;i++) {
PRAGMA(HLS PIPELINE II=EXTERNAL_DATA_WIDTH)
    struct packed_data real_values=input[i];
    for (unsigned int j=0;j<EXTERNAL_DATA_WIDTH;j++) stream.write(real_values.data[j]);
  }
  if (base_its * EXTERNAL_DATA_WIDTH < num_its) {
    struct packed_data real_values=input[base_its];
    remainder_load_data_loop:
    for (unsigned int j=0;j<num_its - (base_its * EXTERNAL_DATA_WIDTH);j++) stream.write(real_values.data[j]);
  }
}

/**
 * Very simple compute stage, simply adds a predefined value to each input value and streams out
 */
static void do_compute(hls::stream<DATA_TYPE> & input_stream, hls::stream<DATA_TYPE> & output_stream, DATA_TYPE add_val, int num_its) {
  main_compute_loop:
  for(unsigned int i=0;i<num_its;i++) {
#pragma HLS PIPELINE II=1
    output_stream.write(input_stream.read() + add_val);
  }
}

/**
 * Writes result data to external memory, for efficiency will write in chunks
 */
static void write_data(hls::stream<DATA_TYPE> & result_stream, struct packed_data * out, int num_its) {
  const unsigned int base_its=num_its / EXTERNAL_DATA_WIDTH;
  main_write_data_loop:
  for (unsigned int i=0;i<base_its;i++) {
PRAGMA(HLS PIPELINE II=EXTERNAL_DATA_WIDTH)
    struct packed_data real_values;
    for (unsigned int j=0;j<EXTERNAL_DATA_WIDTH;j++) {
      real_values.data[j]=result_stream.read();
    }
    out[i]=real_values;
  }
  if (base_its * EXTERNAL_DATA_WIDTH < num_its) {
    struct packed_data real_values;
    remainder_write_data_loop:
    for (unsigned int j=0;j<num_its - (base_its * EXTERNAL_DATA_WIDTH);j++) {
      real_values.data[j]=result_stream.read();
    }
    out[base_its]=real_values;
  }
}
