# belief propagation
1;5202;0c1;5202;0c
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
![img3](https://github.com/clearlycloudy/belief/blob/master/tests/img3.png?raw=true =250x)
### After
![img3after](https://github.com/clearlycloudy/belief/blob/master/tests/out_img3.png?raw=true =250x)

### Original
![img2](https://github.com/clearlycloudy/belief/blob/master/tests/img2.png?raw=true)
### After
![img2after](https://github.com/clearlycloudy/belief/blob/master/tests/out_img2.png?raw=true)

### Original
![img0](https://github.com/clearlycloudy/belief/blob/master/tests/img0.png?raw=true =250x)
### After
![img0after](https://github.com/clearlycloudy/belief/blob/master/tests/out_img0.png?raw=true =250x)