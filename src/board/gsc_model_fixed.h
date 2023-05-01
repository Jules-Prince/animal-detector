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

typedef number_t max_pooling1d_6_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d_6(
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

typedef number_t conv1d_6_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d_6(
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


const int16_t conv1d_6_bias[CONV_FILTERS] = {-69, 12, 76, -30, 9, 108, 71, -22, 58, -40, -25, 37, -23, 65, 3, 32, 43, 33, -28, 12, 16, 5, 35, -30, 15, 12, -10, 89, -20, 54, -14, 35, -19, 96, -68, 49, 55, 32, -23, -16, -2, 36, -22, -53, 109, 37, -2, 53, 12, 35, 15, 55, 18, 16, 50, 68, 11, 46, 34, 49, -3, 36, -31, 112}
;

const int16_t conv1d_6_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{19, 59, 44, -15, -27, 14, -34, -5}
}
, {{-34, -75, 61, -56, 43, 38, 13, 62}
}
, {{-141, -157, 57, 133, -168, -5, 82, 32}
}
, {{-66, 80, 13, 68, -40, -38, 72, -5}
}
, {{13, 30, 15, 4, -49, -15, -35, -43}
}
, {{-101, 14, 42, -96, 1, 37, 27, -75}
}
, {{-45, -56, -95, 8, 13, 20, 10, -146}
}
, {{64, -44, -53, 27, 72, -75, 50, 63}
}
, {{14, -171, 2, -21, 45, -37, -1, 53}
}
, {{3, 19, 67, 2, 17, -47, 22, 25}
}
, {{16, 50, 56, 13, -64, 37, 63, -1}
}
, {{-44, 58, 6, 47, -38, 42, 48, 16}
}
, {{55, 21, 15, -34, 38, 22, -52, -17}
}
, {{-42, 1, 36, 30, 55, -62, 47, -6}
}
, {{-117, 26, -86, 51, 58, 1, 14, 113}
}
, {{-105, 100, 115, -66, 9, -120, -123, -67}
}
, {{-69, 10, 19, 44, -32, -8, 87, 60}
}
, {{30, -4, 34, -19, 57, 17, -78, -39}
}
, {{50, 15, -48, 14, 38, -45, -33, 51}
}
, {{-41, 18, -2, 42, 35, 18, 27, 6}
}
, {{-28, 83, -94, -36, 33, 83, -14, -11}
}
, {{5, 29, -1, -50, -26, 4, 19, 44}
}
, {{50, -89, -65, -9, 21, 53, -23, 25}
}
, {{39, -49, 8, 85, -46, 36, 33, -61}
}
, {{20, -65, 34, 24, -82, 8, -8, -143}
}
, {{-70, -73, -28, -91, -58, -12, -116, -72}
}
, {{9, -22, 89, -63, -53, 49, 42, -24}
}
, {{43, -23, 81, -35, 30, 75, -145, -134}
}
, {{-27, 23, 61, 41, -16, -27, 55, -49}
}
, {{-23, -22, -68, 10, 73, 16, -4, 77}
}
, {{-86, -65, -131, 0, -23, 6, -50, -71}
}
, {{11, -152, 0, -51, 14, -31, -150, 29}
}
, {{-103, 5, -48, 38, 51, 34, -26, 92}
}
, {{-43, -102, -106, -52, 20, 59, 2, 58}
}
, {{7, -53, 40, -15, -37, 52, 54, 31}
}
, {{47, -48, -70, 37, 12, 31, 44, 33}
}
, {{-27, -68, 8, 93, 21, 63, 8, 92}
}
, {{-120, -24, 82, 10, -36, -57, 102, 33}
}
, {{24, -14, 30, 21, -17, -40, 51, 3}
}
, {{-45, 18, -47, -1, 43, 52, 56, 32}
}
, {{21, 67, 30, -16, -39, 57, -69, -6}
}
, {{-46, 20, 0, 48, -42, -3, 52, 80}
}
, {{-40, -58, 30, 45, 36, 62, -43, 0}
}
, {{-70, 8, -56, 46, 40, 53, 28, 25}
}
, {{17, 77, -102, -104, -5, 8, -72, 14}
}
, {{15, 23, -7, 25, 36, 62, -56, -64}
}
, {{-39, -41, 0, 44, 12, 51, 76, -9}
}
, {{22, -21, -114, -22, -81, -71, -104, 91}
}
, {{45, -63, -45, 13, 59, 23, 63, 31}
}
, {{14, 69, -23, -57, 56, -9, 19, -59}
}
, {{-68, 66, 92, -51, -39, -165, -113, -44}
}
, {{32, -36, 47, -85, -74, -11, 30, 33}
}
, {{47, 12, 14, 14, -26, 56, 7, -23}
}
, {{54, 66, -47, -13, 63, -85, 22, 11}
}
, {{24, -129, 50, 57, -72, -23, 71, -79}
}
, {{32, 87, -40, -19, 19, -54, -39, -69}
}
, {{53, -76, 31, 46, -60, 29, 19, -12}
}
, {{2, 70, -86, -46, 28, 1, 4, 35}
}
, {{-15, -64, 56, -40, 37, 6, 71, -50}
}
, {{52, -70, -20, 3, 26, -9, 10, -61}
}
, {{5, 58, -68, -86, 10, 41, 26, 77}
}
, {{34, 60, -23, 5, 45, -48, 0, 20}
}
, {{-15, -97, 45, 84, 106, -24, 54, -20}
}
, {{19, -80, 34, 50, -26, -10, -92, -94}
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

//typedef number_t *flatten_2_output_type;
typedef number_t flatten_2_output_type[OUTPUT_DIM];

#define flatten_2 //noop (IN, OUT)  OUT = (number_t*)IN

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

typedef number_t dense_4_output_type[FC_UNITS];

static inline void dense_4(
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


const int16_t dense_4_bias[FC_UNITS] = {-29}
;

const int16_t dense_4_kernel[FC_UNITS][INPUT_SAMPLES] = {{-22, -8, 1, -41, -24, 40, 41, 24, 20, 1, -39, -4, -15, 14, 0, 0, -19, 45, 16, -8, 9, 39, 3, -2, 26, 28, -73, -21, 67, -58, -39, 11, -6, -17, 19, 22, -20, -39, -30, -60, 33, 33, -14, 37, 1, -48, -5, 40, 57, 148, -13, -35, -64, 25, -16, 32, -128, 49, -38, 29, -62, 24, 27, -78, -26, 89, 54, 40, -68, -47, 71, -135, -140, 117, 100, 28, -77, 11, 66, 38, -13, 11, -17, -12, 5, -12, -20, -34, 24, 28, 19, 17, -10, -1, 50, -35, -25, -6, 14, 64, 12, 7, -28, 56, -24, 5, 20, 17, -15, -11, 30, 10, 58, 63, 10, 35, 19, -53, 26, 33, 51, 103, 45, -58, -119, -109, 17, 23, 0, 65, -9, 31, -65, -44, 66, -38, 27, -33, 56, -32, -23, 52, -82, -73, 33, 10, 27, -81, 18, 32, 55, -57, 31, 73, -90, -48, 1, -31, -81, -9, -27, -37, 61, -15, 39, 35, -60, 62, -1, -58, -25, -41, -40, -30, -76, 66, 93, 82, -56, 24, -80, -161, 3, -16, -24, -30, 13, -30, 12, 37, 7, -10, 19, -17, 11, 12, -2, 1, -20, -21, 16, 69, 8, 39, 19, -46, 71, 42, 70, -118, -25, -6, 53, 10, 0, -46, 44, -45, -30, -80, 0, 99, 2, 24, -99, 55, -21, -127, 84, 118, 64, -76, -118, 49, -11, -1, -6, -8, -5, 1, -3, 18, 7, -29, 26, 28, 24, -20, -42, 16, 16, -25, -39, 6, 35, 26, 12, 0, 12, -6, -20, 19, -17, 12, 10, -20, 16, 24, -16, 9, -28, -10, 43, 4, -13, -23, 18, -2, -35, 14, 11, -22, 34, -12, -38, 25, 20, 1, 2, -3, -12, 19, -9, -40, 11, 3, -13, 1, 26, -25, -43, -45, -14, 19, -18, -9, 16, -5, 11, -1, -14, 5, 16, 14, 6, -33, -7, 23, 15, 13, -26, 44, -13, -8, 7, 25, 43, 6, -55, -11, 37, -40, -15, 11, -14, 37, -15, -11, -24, 34, 10, 2, -5, 15, 11, 8, 28, 16, 16, -8, -26, -28, -69, 32, -11, -50, -29, -29, 10, 7, -17, -46, 28, 6, -20, 103, -79, -5, 22, -38, 0, 3, 4, 43, 7, -70, 8, -84, -10, 42, -37, 22, 33, -55, -74, -35, 10, 30, 145, 41, 145, -41, -24, -80, 107, 3, -42, 23, 104, -132, 5, 13, -36, -12, -3, 11, 130, 10, -96, -30, -89, -108, 76, -56, 106, 21, -3, 38, -6, 30, 20, 16, -15, 6, 14, 29, -7, 26, -26, -30, -53, -18, 28, -10, -5, -30, -19, -16, 10, -6, -14, -22, 22, 10, 16, -27, 26, -2, -18, -45, 6, 21, 26, -15, -38, -20, -2, 29, -7, -34, -25, -20, -40, -4, 0, 45, 1, -50, -21, 15, 18, 15, 1, -50, 11, 18, 10, 16, 30, 0, -35, -7, 24, 0, 3, -67, 33, 12, 4, 35, -5, -7, -8, 49, -17, 1, 7, 11, -28, -29, -15, -31, 14, 21, 18, -7, -28, -26, 9, 19, 15, -23, -10, 18, 20, -4, -7, -7, 8, 14, 92, 69, -3, 10, -76, 16, -26, -10, 5, -47, 31, 13, -13, 32, -62, -3, 29, 6, 15, -15, -64, 25, -3, 20, 72, -120, -18, -25, -5, 8, -20, 19, -5, -6, -33, 13, 17, -25, 39, 7, 15, 21, 11, -24, 29, 11, -5, 2, 28, 38, 7, -55, 17, 12, 38, -34, -7, -10, -26, 56, -22, -4, 14, -22, -30, -4, 87, -43, -5, 56, -149, 69, -15, 42, -7, -86, -1, -77, -10, -10, 17, 26, 23, -21, -33, -5, -11, -38, 7, 35, -38, 57, -23, -8, 19, -17, -27, 38, 24, -19, 56, 56, 6, 56, 34, 14, -31, 8, 33, 6, -39, 79, -49, 16, -9, 28, 23, -47, 4, 20, -68, -3, 66, -88, -36, 12, -129, 110, 17, -35, 15, -33, -129, -4, 16, 3, -3, -16, -9, -35, -34, -43, -28, -9, -80, -42, -30, -57, 0, 10, 26, 32, -84, -129, -47, -61, -9, -28, 21, -58, -25, -7, 10, -41, -21, -12, -13, 6, 96, 4, -9, 0, 10, 86, 2, -12, 50, -46, 42, 61, -45, 3, -5, 72, 0, 16, 16, -18, -68, -10, 36, 47, -13, -53, 99, -61, 99, 71, 0, 0, -161, -105, -53, -53, -24, 92, -23, -78, 22, -1, 22, 23, 38, 34, -39, 29, 20, -36, 25, 30, 45, -5, 3, -27, 55, -22, -18, 1, -6, -17, 13, 0, 7, 78, -19, 58, -16, -36, -55, -16, -13, 3, 72, -17, 4, -42, 10, -47, 3, 3, 19, -17, 62, -61, -35, 39, -24, 0, 104, -1, 23, 5, -64, -46, 21, 5, 24, -4, 23, -3, -12, -11, -41, -49, -49, -110, -72, -97, -104, 8, 28, 54, -132, -81, -72, -101, 75, -33, -36, 20, 63, -13, -68, -57, 57, -40, 0, -73, 3, 8, -4, -4, -46, -39, -8, -56, 108, 66, 24, -103, -154, -55, -92, 68, -39, 0, 5, 17, -14, -12, 15, 27, 39, -26, 23, -62, 35, 51, -25, -16, 16, -41, -86, -32, 38, 64, 68, 20, 18, -1, -76, -49, 28, 41, 6, 20, 59, -29, -42, -22, 20, 24, -13, -10, 3, 21, -33, -13, 21, 61, 72, 31, 28, -73, 38, -23, 39, 10, -37, 32, -6, -12, -8, 20, 20, -30, 18, 8, 20, 21, 32, 70, -39, -4, -16, 7, 47, -5, -17, -1, 20, -10, -38, -16, 2, -52, 3, 21, 38, -43, 12, -27, -13, -6, 5, -42, -3, -40, -53, 25, -7, -4, 45, -41, 50, 33, -110, 16, 33, 13, -15, -32, -6, 42, 24, 20, 6, 11, -4, -62, -4, -10, 23, 9, -28, -29, 6, 35, -7, -19, 0, 16, -94, 65, -29, 51, -49, -13, -45, -38, 5, 0, -13, -13, -33, -14, 2, 28, 22, -85, -24, -80, -100, -51, 75, 38, -70, -60, -4, -15, -7, -23, 24, -10, -11, 41, -2, -27, 6, -6, 23, -7, -8, -13, 60, 5, -24, 16, 29, 0, 38, -31, -36, 48, 34, 29, -22, 14, 5, 12, -1, 27, 22, 30, 24, 12, -10, 19, 5, 31, 17, -6, 52, 22, -22, 26, 32, 58, 27, -3, -28, -14, 19, -26, -6, -10, 49, 13, 13, 24, -33, 5, 22, 16, 35, 15, 5, 44, -2, -62, 25, 20, -27, 50, -30, -34, 33, 48, -25, -12, 38, 11, -17, -2, -31, 11, 8, -19, -8, -29, -3, -19, 16, -36, 0, -8, -28, 27, 25, -20, 0, -9, -29, 10, 34, 17, 26, -11, 25, 2, 8, 68, 16, 11, 29, -4, 47, -49, 18, 38, 20, -15, -16, -27, 0, 44, -7, -6, -63, 64, -30, 55, 25, 11, -15, 29, 16, 5, 30, 13, 18, -7, -4, 22, 12, 13, 20, 22, -59, -32, 21, 33, 31, 28, -62, 53, 45, 17, -27, 11, -41, 4, 14, 41, -19, 4, 40, 120, -39, 12, 73, -87, 134, 91, 5, -24, -197, -63, -10, -55, -25, 13, 21, 27, 11, -3, 14, -36, -27, 26, -34, -22, -25, 16, 4, -22, -5, -26, 20, 0, 10, -41, -24, -8, -17, -31, -47, 5, 40, 29, 3, 5, -8, 29, 54, 40, -5, -43, -37, 41, -34, 8, 35, 17, 31, 26, -7, 1, 7, 40, 21, 65, -50, -23, 78, 1, 61, -34, 22, -6, 40, 20, -21, -9, 34, -22, -2, -76, 40, 58, 12, 104, 128, 46, 32, -139, -36, 28, 5, -1, -6, -19, -9, -37, 26, 18, 41, -37, 0, -30, 12, 4, -16, 22, 12, 17, -8, 19, -28, -47, 36, 25, 38, -4, -20, -11, 34, 23, -22, -13, 6, 18, 12, -46, 12, -48, -15, 112, 8, -4, -88, -35, 56, -28, -44, 21, -5, -64, -14, -21, 114, -13, 32, -63, 44, 21, 25, -5, 94, -102, 27, 43, -45, -25, -21, 10, 99, 97, -22, -36, -90, -31, -6, -21, 131, 28, 16, 21, 14, -15, -16, 8, 4, 11, -11, -41, -16, 4, 49, -28, 16, 35, -172, 84, 42, 71, -13, -85, 18, -97, -87, -63, -9, 29, -5, -40, -30, 3, -22, 5, -15, -26, -5, -17, 10, 29, -10, -52, -30, -22, -53, 12, -25, -22, 1, 34, -3, 19, 18, -15, -7, 7, -3, -22, 13, 1, -32, -5, 1, -35, 25, 25, -30, 30, -39, -10, 40, 18, -50, 8, 40, -63, 50, -3, 37, -77, -48, 66, -88, -21, -42, 57, -100, -12, 36, -12, 32, 86, -137, 8, -20, -13, 16, 68, 13, 26, -40, 74, -47, 89, -8, 22, 34, -14, -62, -27, -31, -26, 37, 99, 34, 0, 12, 81, 21, 42, -24, 42, 35, -87, -145, -36, 16, -29, 29, 24, 20, -26, 9, 26, -27, -36, -25, 15, -7, -23, 13, 11, 14, 36, 32, 2, 9, 27, -53, 61, 16, -23, 30, 5, -23, 76, -44, -28, -24, 24, -84, 10, -13, -9, -20, -18, -30, -16, 11, 75, -12, 12, 20, -55, 90, 11, -56, 33, -60, -32, 0, -71, -68, -58, 11, -69, -51, 17, -16, -40, 6, 46, -25, 7, -22, -5, -54, -13, -4, -23, -46, -40, 19, -43, -30, 64, -101, 45, 25, 14, 2, -15, 39, -79, 48, -32, -57, 2, 29, 2, -16, 58, 109, 46, -34, 78, -64, 102, -31, -62, 40, -9, 4, -66, -25, -5, 19, 9, -29, 0, 7, 58, -46, -16, -8, -22, 12, 8, 11, 18, 56, -21, 31, 11, -26, 63, -55, 57, -37, -78, -19, -30, 1, 30, -13, -29, 3, -15, -2, 8, -16, -26, -19, 2, -33, -6, -3, -49, 35, -30, -4, -5, -9, -28, 18, -11, -148, -22, 45, -16, -57, 19, 8, 4, 5, 51, 8, -19, 35, 48, -68, 44, 8, -3, 9, 57, 20, -58, 14, -30, 20, 60, 59, 68, -75, 74, 25, -34, -10, 103, -143, -22, 53, -19, 40, -61, -18, 142, 114, 15, 54, -11, -54, -17, -135, 39, -35, -68}
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
  //dense_4_output_type dense_4_output);
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
#include "max_pooling1d_6.c" // InputLayer is excluded
#include "conv1d_6.c"
#include "weights/conv1d_6.c" // InputLayer is excluded
#include "flatten_2.c" // InputLayer is excluded
#include "dense_4.c"
#include "weights/dense_4.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_4_output_type dense_4_output) {

  // Output array allocation
  static union {
    max_pooling1d_6_output_type max_pooling1d_6_output;
  } activations1;

  static union {
    conv1d_6_output_type conv1d_6_output;
    flatten_2_output_type flatten_2_output;
  } activations2;


  //static union {
//
//    static input_5_output_type input_5_output;
//
//    static max_pooling1d_6_output_type max_pooling1d_6_output;
//
//    static conv1d_6_output_type conv1d_6_output;
//
//    static flatten_2_output_type flatten_2_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  max_pooling1d_6(
     // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_6_output
  );
 // InputLayer is excluded 
  conv1d_6(
    
    activations1.max_pooling1d_6_output,
    conv1d_6_kernel,
    conv1d_6_bias,
    activations2.conv1d_6_output
  );
 // InputLayer is excluded 
  flatten_2(
    
    activations2.conv1d_6_output,
    activations2.flatten_2_output
  );
 // InputLayer is excluded 
  dense_4(
    
    activations2.flatten_2_output,
    dense_4_kernel,
    dense_4_bias, // Last layer uses output passed as model parameter
    dense_4_output
  );

}
