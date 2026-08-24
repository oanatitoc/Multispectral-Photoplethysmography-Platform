#ifndef PPG_FIRMWARE_VARIANT
#define PPG_FIRMWARE_VARIANT 0
#endif

#if PPG_FIRMWARE_VARIANT == 0
#include "firmware/fast_6ch.hpp"
#elif PPG_FIRMWARE_VARIANT == 12
#include "../lab/tcs3448_red_nir_12ch_lab/src/main.cpp"
#elif PPG_FIRMWARE_VARIANT == 18
#include "../lab/tcs3448_18ch_lab/src/main.cpp"
#else
#error "Unknown PPG_FIRMWARE_VARIANT. Use 0, 12, or 18."
#endif
