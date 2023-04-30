/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 100
#define FC_UNITS 1


const int16_t dense_24_bias[FC_UNITS] = {37}
;

const int16_t dense_24_kernel[FC_UNITS][INPUT_SAMPLES] = {{-20, -59, -54, 65, 60, 33, 105, -70, -75, 9, 24, 48, -108, 48, 85, 77, 52, 73, -73, -42, -79, -22, -56, -34, -91, 34, 198, -19, -61, 97, 55, 74, 0, -3, 37, 2, 22, -100, -53, -118, 63, -101, -96, -74, -43, 71, -122, -40, -26, 65, 44, 43, -65, -22, -14, -41, 46, 23, -133, -71, -13, 139, -139, -15, -96, -171, -85, -121, 29, 96, -81, 6, -35, 42, -200, -36, -118, -132, -72, 6, 0, -20, 95, 10, 0, -57, -33, 96, -153, -73, -88, -109, -32, -41, -52, -12, 106, 58, 209, -63}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS