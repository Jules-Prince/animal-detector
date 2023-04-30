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
