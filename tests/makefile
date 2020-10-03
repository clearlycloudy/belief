release:
	clang++ -std=c++1z -O3 -o out.o img_denoise.cpp ../bp.cpp lodepng.cpp -pthread

release_gpu:
	clang++-10 -std=c++17 -O3 -x cuda --cuda-gpu-arch=sm_75 -o out_gpu.o img_denoise_gpu.cpp lodepng.cpp -pthread -L/usr/local/cuda-10.2/lib64 -lcudart

debug_gpu:
	clang++-10 -std=c++17 -g -x cuda --cuda-gpu-arch=sm_75 -o out_gpu.o img_denoise_gpu.cpp lodepng.cpp -pthread -L/usr/local/cuda-10.2/lib64 -lcudart
