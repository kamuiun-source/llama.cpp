#pragma once

#ifdef GGML_USE_HIP
void ggml_hip_gfx906_set_iq2s_signed_lut(bool enabled);
int ggml_hip_gfx906_iq2s_signed_lut();
#endif
