/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 8
#define FC_UNITS 1


const int16_t dense_1_bias[FC_UNITS] = {-74}
;

const int16_t dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{397, -235, -129, 139, -228, -341, -88, -114}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS