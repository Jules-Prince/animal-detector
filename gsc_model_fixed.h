#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>

#define FIXED_POINT	9	// Fixed point scaling factor, set to 0 when using floating point
#define NUMBER_MIN	-32768	// Max value for this numeric type
#define NUMBER_MAX	32767	// Min value for this numeric type
typedef int16_t number_t;		// Standard size numeric type used for weights and activations
typedef int32_t long_number_t;	// Long numeric type used for intermediate results

#ifndef min
static inline long_number_t min(long_number_t a, long_number_t b) {
	if (a <= b)
		return a;
	return b;
}
#endif

#ifndef max
static inline long_number_t max(long_number_t a, long_number_t b) {
	if (a >= b)
		return a;
	return b;
}
#endif

#if FIXED_POINT > 0 // Scaling/clamping for fixed-point representation
static inline long_number_t scale_number_t(long_number_t number) {
	return number >> FIXED_POINT;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) max(NUMBER_MIN, min(NUMBER_MAX, number));
}
#else // No scaling/clamping required for floating-point
static inline long_number_t scale_number_t(long_number_t number) {
	return number;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) number;
}
#endif


#endif //__NUMBER_H__
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  1
#define INPUT_SAMPLES   100
#define POOL_SIZE       4
#define POOL_STRIDE     3
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_4_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_4(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       33
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    8
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_3_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_3(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    1
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  8


const int16_t conv1d_3_bias[CONV_FILTERS] = {77, 55, -70, 62, -43, 23, 120, 25, -79, 86, 53, -4, 96, 51, 27, 98, -6, 90, 102, -6, 102, 110, -41, 35, 91, -15, -20, -41, 169, -156, -63, 103, -89, 102, 16, 134, -67, -26, 98, 93, 19, 126, 109, 104, 120, -42, 84, 70, 57, -56, -27, 87, -69, -42, 4, -48, 63, -118, -51, 53, -55, -53, 114, 77}
;

const int16_t conv1d_3_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{34, 26, -112, -182, -226, -137, -29, 99}
}
, {{-130, -151, -174, -223, -17, -94, -182, -156}
}
, {{-1, 24, 60, 17, 13, 72, 110, 4}
}
, {{49, 45, 58, 5, 58, 49, 21, 21}
}
, {{-141, 86, -111, 55, -36, -26, 160, 141}
}
, {{-16, 114, 131, 51, -151, -158, -189, -209}
}
, {{35, -142, 46, 38, -57, 13, 34, 8}
}
, {{45, 60, 61, -34, -28, -45, 20, 103}
}
, {{19, 73, -6, -46, 48, 4, 0, 32}
}
, {{-127, -41, -44, 5, -22, 99, 94, 152}
}
, {{9, 43, -23, 30, 21, 63, -31, 43}
}
, {{7, 70, -15, 28, 62, -31, -65, -28}
}
, {{43, -124, 6, 96, 64, -18, -100, 39}
}
, {{75, -124, -51, 21, 77, -45, 47, 38}
}
, {{36, 23, -43, 109, 87, 29, -88, -99}
}
, {{-128, 106, 54, -56, -60, -31, 46, 134}
}
, {{-50, -20, 40, 68, 27, 58, -9, 36}
}
, {{-98, -7, 75, 107, 31, -110, 64, 41}
}
, {{80, -96, -56, 60, -33, -28, 90, 2}
}
, {{-80, -8, -76, 31, 95, 11, 56, 38}
}
, {{57, -89, 14, 16, -42, 39, 54, -36}
}
, {{49, -86, -32, -2, 9, 58, 53, 27}
}
, {{10, -121, 3, 12, 56, 52, 0, 77}
}
, {{24, 80, 41, -63, 47, -17, 0, -39}
}
, {{-329, 35, -116, 27, -26, 40, 92, 111}
}
, {{-23, -51, 41, -20, 64, 94, 88, 117}
}
, {{15, 5, -36, 48, 64, 83, 69, -77}
}
, {{29, 79, -44, -3, 52, 43, -52, -44}
}
, {{22, 85, -41, -151, -128, -73, -11, 73}
}
, {{14, 17, 84, 44, 37, -55, 28, 136}
}
, {{-10, 15, -19, 99, 53, -42, 81, 33}
}
, {{19, -90, 71, 119, -8, -25, -116, -85}
}
, {{-26, -101, 163, 142, 11, -72, 133, -37}
}
, {{-30, -9, 97, -60, -55, -26, 63, 38}
}
, {{-137, -166, 31, -75, -130, -21, -120, -214}
}
, {{74, -151, -21, 21, 14, 37, 49, 29}
}
, {{-59, 27, -11, 40, 47, 39, 3, 43}
}
, {{27, -48, -74, 23, 32, 21, 85, 95}
}
, {{-319, 16, -119, 54, -30, 37, 61, 126}
}
, {{30, 10, -80, -92, -65, -20, 131, 168}
}
, {{-34, -75, -20, 57, 73, 60, 64, 103}
}
, {{-159, -21, -93, 94, 54, 7, 138, 115}
}
, {{7, 81, -147, -83, 30, 11, 73, 98}
}
, {{88, -114, -77, 75, -34, 30, 99, -12}
}
, {{29, -63, 76, -87, -112, 98, 45, 151}
}
, {{5, 27, 55, -6, -49, 77, 71, -88}
}
, {{42, 24, -64, 17, -26, 10, 2, 50}
}
, {{18, 74, -36, -65, 41, -9, -49, 4}
}
, {{35, 3, 23, 63, -33, 82, -32, 19}
}
, {{-16, -53, -7, 79, 53, 61, 20, 64}
}
, {{-80, -41, 41, 86, 103, 23, 19, 27}
}
, {{-15, -58, -172, -56, -55, -65, -86, -162}
}
, {{-16, 0, 29, -27, 21, 89, 61, 26}
}
, {{21, 8, 51, -45, -24, 151, 110, -112}
}
, {{47, -87, -52, 29, 12, 43, 100, 14}
}
, {{4, 0, 77, -60, -14, 59, 16, -28}
}
, {{0, 7, 77, 58, -53, -26, 28, -61}
}
, {{22, 71, -24, 25, 37, -32, -14, 80}
}
, {{-11, -28, -45, 130, 67, 47, 74, -44}
}
, {{-66, -191, 44, -1, -68, 17, -155, -151}
}
, {{-76, -5, 39, 82, 12, 42, 31, 77}
}
, {{65, 6, 104, 133, -163, -8, 152, -121}
}
, {{-92, -86, 65, -37, 49, 70, -31, 119}
}
, {{-178, 56, 149, -135, -7, -15, -48, 76}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_DIM [26][64]
#define OUTPUT_DIM 1664

//typedef number_t *flatten_7_output_type;
typedef number_t flatten_7_output_type[OUTPUT_DIM];

#define flatten_7 //noop (IN, OUT)  OUT = (number_t*)IN

#undef INPUT_DIM
#undef OUTPUT_DIM

/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_SAMPLES 1664
#define FC_UNITS 1
#define ACTIVATION_LINEAR

typedef number_t dense_26_output_type[FC_UNITS];

static inline void dense_26(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]) {			                // OUT

  unsigned short k, z; 
  long_number_t output_acc; 

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0; 
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ( kernel[k][z] * input[z] ); 

    output_acc = scale_number_t(output_acc);

    output_acc = output_acc + bias[k]; 


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = clamp_to_number_t(output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0)
      output[k] = 0;
    else
      output[k] = clamp_to_number_t(output_acc);
#endif
  }
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 1664
#define FC_UNITS 1


const int16_t dense_26_bias[FC_UNITS] = {-63}
;

const int16_t dense_26_kernel[FC_UNITS][INPUT_SAMPLES] = {{-127, 24, 75, 89, -13, 40, 45, 14, -14, 73, 49, 52, 23, 13, -51, -176, -88, 103, 136, 155, 96, -111, -171, -278, -131, -53, -10, -44, -232, -19, 40, 14, -47, -12, -33, -51, -74, -7, 27, -20, -71, -160, -109, -110, 26, 39, 13, 49, -52, -152, -214, -195, 7, -26, -4, 10, 16, -23, 14, 21, 7, -26, -31, -23, -4, -8, -41, -57, 9, 9, -36, 26, 28, 0, 61, 14, -3, -1, 4, -30, -19, -41, -26, -23, -8, 0, -35, 9, -39, -35, 3, 23, -15, -55, -73, -44, -34, -1, -25, -5, -28, -10, -3, -28, 161, 201, 5, 68, 52, -56, -18, -44, -36, 36, 6, -20, -7, 11, -31, 58, 53, -67, -4, -39, -148, 124, 104, 22, 26, -132, 162, 44, 10, 23, 128, 0, 68, 174, 147, 112, -11, 26, -2, -215, 75, 157, 264, 221, -70, -134, -163, -200, -50, 109, 183, 73, 71, -111, -67, 92, -103, -137, -19, -20, -79, -39, 37, -184, -45, 38, 18, -32, -87, 10, -111, 17, -44, -64, -42, -92, -174, 23, -2, -21, -35, -21, -38, 9, 39, 29, -8, 21, 7, 11, 42, -58, -59, -61, -21, 15, 14, 3, 26, 50, -22, -31, -24, -2, -10, 11, -15, 28, 8, 4, 7, 2, 0, 19, -20, -16, 13, 58, -9, 6, -18, 39, 47, -48, 14, 93, -21, 43, 60, -7, -110, 94, -7, -42, -5, 17, 4, 67, 89, 57, 35, -54, -82, -23, -17, -7, -8, 15, -37, -62, -61, 0, 154, 123, -7, -55, -16, 4, -39, -19, -16, 41, 0, -50, -7, 9, -50, -31, 30, -11, -15, -8, -48, 1, 15, -43, -29, -8, -13, -9, 35, -11, -17, -10, 23, -9, 46, 4, 13, -23, -10, 68, 0, -54, -39, 56, 62, 110, -12, -17, 56, -67, -44, 10, -14, 21, 61, -38, 43, -87, 55, -19, 50, -36, 17, -25, -56, 7, 72, -28, 58, -80, 36, 52, 33, -60, -73, -12, -60, -67, -20, 2, -8, 58, 38, -60, 3, -77, -17, -122, 51, 38, 38, -43, 11, -54, 22, 30, 23, -45, -5, 25, -42, 99, -29, 25, 62, -77, 32, 49, -22, -13, 69, 53, 18, 18, -4, -40, -11, 52, -14, -15, -41, 20, 92, 69, -30, -65, 9, 18, -1, -9, -11, 10, 6, 35, 90, 76, -24, -21, 82, -75, -18, -75, -71, -49, -6, 41, 26, -66, -7, 28, -38, -6, -38, -20, 18, 25, -43, -109, -5, -111, -49, 34, 13, 19, -4, 13, 13, -29, 1, 67, 31, 24, 19, -47, -13, -50, -20, -2, 16, 2, 47, -10, -51, 1, 36, 48, -63, 55, -61, 32, -50, 33, -21, -25, 41, -6, 43, 46, 8, -53, -123, -2, -23, -5, -26, -7, -6, 5, -33, -75, 9, 18, 22, -63, -75, -48, -8, -113, 4, -9, 16, -89, 27, -53, -44, 23, 27, -72, 1, -11, -87, 125, 5, 5, 85, -136, -52, 85, -31, 84, 14, 71, -42, 13, -4, -22, 53, 56, 68, -40, 4, -37, -59, 43, 22, 9, 53, -21, -2, -7, 15, 22, 115, 63, 20, -51, -5, -42, 32, -62, -51, 4, -51, -39, -23, -26, -67, 12, 28, -60, -8, 29, -47, 87, -50, -48, 85, -15, -95, 47, 5, -70, 7, -79, 4, -54, 23, 39, 28, -6, -4, -20, 10, 13, -10, -75, -33, -36, -46, 68, -18, -4, 37, -20, -33, -27, 87, -34, 110, 15, -39, -49, 27, 48, 47, 24, -25, 34, 12, -75, 69, -7, 31, 57, 0, -21, -63, 1, 101, 73, 78, 127, -27, -34, -27, -10, -13, 6, 18, -28, 5, 27, -35, -51, 25, 56, 9, 6, -52, 37, 31, -62, -42, 11, -28, 13, -32, -27, 12, 241, -185, -98, -6, 73, 99, 33, -8, 56, -10, -100, -97, -28, 64, 51, 88, -100, 5, -64, -205, 85, 204, 87, -27, -158, 64, 29, 46, -9, -22, -2, -23, -2, 39, 5, 32, -2, -38, -76, -45, -11, 8, 45, -26, -13, -22, 38, 16, 38, 28, 23, -21, -4, 24, 4, 5, 15, -22, 21, 50, 5, 4, -43, -47, 16, 34, -30, -3, -9, 22, 4, 1, -41, 43, 9, 17, 34, -13, 15, -5, 7, -40, 24, 25, -38, 14, 43, -1, -3, -18, 76, 83, 37, -66, -27, 106, -58, -14, 36, 50, 56, 15, -36, -58, 41, 58, 53, -57, 49, 20, 24, 7, 27, -29, 78, 92, 94, 33, 11, -35, 132, 201, 137, 3, -36, -119, -115, -95, -47, 34, -47, -27, 26, 20, 21, -26, 10, -46, -28, -2, 45, 2, -43, 0, 47, -16, -20, -9, 54, 33, 78, 7, -51, 0, 4, -57, 2, -12, 47, 0, -5, 32, 24, -5, -8, 45, 29, 51, -16, -60, -5, 19, -30, 61, 20, -13, 38, -1, -29, 68, 60, 71, -51, -73, 118, 9, 1, 5, -7, -121, -7, 154, -75, -46, -10, -16, 182, 140, -76, -116, 11, -103, -126, -78, -76, -90, -5, -245, 10, -20, 3, -50, 26, -56, 34, 63, 24, -15, 71, 51, 96, -89, 26, 108, -71, -25, 82, 59, -34, -19, -39, 20, 154, -72, 21, -4, -51, 37, 0, -69, -45, 0, -31, -59, 52, -2, -36, -30, -15, -52, 30, -59, -53, 23, -34, 63, -48, -101, -42, -10, -5, -198, 26, 46, 11, -22, 27, -19, 1, -50, -6, 22, -65, -75, -24, -129, -136, 2, -123, 20, 99, -167, -54, -73, -193, 49, -109, -16, -81, 17, -126, 58, 41, 0, -42, -20, -133, -61, 34, 86, -49, 10, 0, -110, 51, 4, -24, 63, -86, -54, 41, 15, 20, -33, 38, 1, -38, 34, -28, 2, 3, 51, 26, 30, -24, 3, 1, -14, -1, 23, 10, 29, 26, -28, 15, 40, 36, 13, -11, 9, 13, 2, -67, 40, 62, 70, -37, 3, 13, 34, -30, -6, 34, 55, 44, -17, 22, 2, 39, 75, -8, 115, -32, -27, 184, -134, -63, -68, 55, 70, 35, 63, 88, -12, -138, -84, -68, 81, 61, 63, 10, 48, -40, -168, 82, 192, 82, 52, -72, 12, -55, -27, 64, 5, 46, 101, 23, -39, -64, -70, -97, 35, 7, 24, 26, 24, -121, 15, 26, 51, 108, -43, -114, -146, -148, -79, 60, 33, -10, -18, -2, -20, 17, 60, 51, 5, 7, -42, -66, -44, -5, -15, 3, -7, -13, -24, 9, 48, 36, 95, 88, -154, 142, -51, 7, -44, 27, -30, 33, 93, 81, -11, -81, -96, -17, -74, 28, 9, -14, 37, -85, -122, 8, 50, 120, 52, 13, -75, -6, -29, 22, -115, 16, 59, 45, -16, -28, -95, -60, 27, 80, -24, 34, 28, -104, 83, 40, -24, 113, -48, 1, 3, -100, 13, -70, -44, -59, 66, -111, 37, 37, 23, -69, 13, -77, -69, 37, 4, -47, 39, -28, -59, 120, -47, -55, 81, -38, 14, 76, 15, -86, 47, -22, 5, -5, 51, 13, 20, -54, -39, -31, 55, -54, 39, -39, -52, 62, -34, -54, 42, -24, 78, -30, -110, -62, -10, -41, 14, -26, 39, -2, 27, 19, 70, -31, -32, 6, 62, 81, 0, -99, 52, 40, -68, -17, 18, 4, 96, 39, -51, 28, 0, -6, -43, 9, 11, -11, -13, -14, -8, -4, -33, -34, 35, 6, 3, -44, -24, -62, 31, 28, -42, 6, -14, -74, -46, -60, -42, -25, 24, -19, -46, 14, -7, -9, -8, 74, -24, -51, 17, 51, -20, 26, -44, -15, 97, -37, -56, -2, -104, -52, 11, -73, -14, 2, -23, -18, 11, -1, -2, -24, -1, 0, -37, -37, -19, -53, 75, -54, 25, -32, -64, -14, -13, -10, 18, -26, 2, 11, -56, 46, -13, 21, 23, 5, -12, 40, 44, 72, 46, -5, 31, -12, 2, 27, -27, 31, 13, 41, -16, -16, 7, 2, 103, 97, -116, -10, 13, 78, 21, -4, 15, 17, 22, 109, 24, 9, 22, -29, -24, 1, -59, -4, 39, 23, -8, -85, -60, 0, 77, 69, -138, -98, -18, -25, 11, -47, -44, -21, -19, -75, -34, 17, -3, -24, -163, -98, -56, -1, 4, 93, 50, -21, -189, -245, -208, -200, 133, 39, 43, -12, 31, -21, -19, 11, 36, 24, 26, 19, 7, -18, 36, -41, 35, 90, -9, -33, 26, 3, 38, 84, 51, 34, -7, -10, 41, -58, 61, 7, -2, -5, 6, -14, -32, 14, -38, 89, 25, -66, 8, 61, -62, 4, 30, 2, 40, 94, -64, 15, 27, -51, 25, -26, 80, -96, 44, 59, 82, -63, 13, -37, -54, 29, 24, 2, 39, 50, -65, 67, 23, -39, 122, 19, 62, 139, -28, -9, 33, -28, 27, -5, 10, -14, 29, 19, -30, 26, 48, 14, -25, -16, -4, 73, -6, -39, 10, 8, 13, 43, 8, 10, -3, -29, -40, -14, 31, 37, -58, 47, 0, -63, 18, 60, 3, 11, 24, -47, 69, 67, -103, -1, -35, -43, 27, -32, -104, 37, 16, -31, -47, 43, 31, -2, 38, 34, -47, 12, 0, 23, 14, 1, -33, 60, 30, -13, 110, -2, 14, 90, 0, -6, 33, -22, -104, 27, 19, 34, 28, -6, 5, 15, -8, -5, 17, -17, 0, -13, -12, -10, 44, -1, 29, -2, -7, -14, 40, -19, 63, 58, 0, -2, -168, 7, 71, -54, -76, 45, -62, 24, -41, -58, -18, -18, -38, -50, -98, -116, -37, -183, 87, 130, -3, -102, -98, -161, -56, 52, 21, 37, -36, 2, 19, -19, -17, 71, 54, 30, 54, -6, -34, 14, 33, -2, 40, 14, 67, 11, -44, 7, 55, 45, 3, 18, -23, -46, 54, -23, -23, 73, 13, -56, 41, 60, 11, 27, 0, -105, 139, -4, -111, 94, -6, -3, 124, -1, -79, 101, -127, -6, -2, -117, -83, 54, -91, 42, 5, 57, -15, -31, -82, -112, -5, 66, -21, 33, 26, -155, -10, 38, 39, 125, -16, -85, -123, 66, -48, -58, 106, -31, -64, -125, 73, -103, -30, 67, -75, 8, 87, 86, 53, -29, -10, -40, 47, -68, 103, -68, -51, -135}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define MODEL_OUTPUT_SAMPLES 1
#define MODEL_INPUT_SAMPLES 100 // node 0 is InputLayer so use its output shape as input shape of the model
#define MODEL_INPUT_CHANNELS 1

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  //dense_26_output_type dense_26_output);
  number_t output[MODEL_OUTPUT_SAMPLES]);

#endif//__MODEL_H__
/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"

 // InputLayer is excluded
#include "max_pooling1d_4.c" // InputLayer is excluded
#include "conv1d_3.c"
#include "weights/conv1d_3.c" // InputLayer is excluded
#include "flatten_7.c" // InputLayer is excluded
#include "dense_26.c"
#include "weights/dense_26.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_26_output_type dense_26_output) {

  // Output array allocation
  static union {
    max_pooling1d_4_output_type max_pooling1d_4_output;
  } activations1;

  static union {
    conv1d_3_output_type conv1d_3_output;
    flatten_7_output_type flatten_7_output;
  } activations2;


  //static union {
//
//    static input_10_output_type input_10_output;
//
//    static max_pooling1d_4_output_type max_pooling1d_4_output;
//
//    static conv1d_3_output_type conv1d_3_output;
//
//    static flatten_7_output_type flatten_7_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  max_pooling1d_4(
     // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_4_output
  );
 // InputLayer is excluded 
  conv1d_3(
    
    activations1.max_pooling1d_4_output,
    conv1d_3_kernel,
    conv1d_3_bias,
    activations2.conv1d_3_output
  );
 // InputLayer is excluded 
  flatten_7(
    
    activations2.conv1d_3_output,
    activations2.flatten_7_output
  );
 // InputLayer is excluded 
  dense_26(
    
    activations2.flatten_7_output,
    dense_26_kernel,
    dense_26_bias, // Last layer uses output passed as model parameter
    dense_26_output
  );

}
