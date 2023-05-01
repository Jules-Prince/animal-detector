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
