# belief propagation
1;5202;0c
BP algorithm in C++, along with some examples.

# Implementation
    - bp.cpp & bp.hpp (CPU version)
    - bp_gpu.hpp (GPU version)
     
# Examples

## Image Filtering
    - tests/img_denoise.cpp
    - tests/img_denoise_gpu.cpp

Using arbitrary settings for tests:

### Original
<img src="https://github.com/clearlycloudy/belief/blob/master/tests/img4.png?raw=true" width="250"/>
    
### After
<img src="https://github.com/clearlycloudy/belief/blob/master/tests/out_img4.png?raw=true" width="250"/>

### Original
<img src="https://github.com/clearlycloudy/belief/blob/master/tests/img2.png?raw=true" width="150"/>
    
### After
<img src="https://github.com/clearlycloudy/belief/blob/master/tests/out_img2.png?raw=true" width="150"/>