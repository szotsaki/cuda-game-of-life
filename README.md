CUDA Game of Life generator
===========================

CUDA Game of Life generator

This application
1. generates a random Game of Life board, then
2. computes the given consecutive steps afterward, finally
3. writes the result into the given file in the format needed by the [viewer](https://github.com/szotsaki/game-of-life-viewer).

The program uses the "infinite field" type of Conway's Game of Life.

Compiling information
---------------------
The following tools and developer libraries are needed to compile this project:

- [CUDA Developer library](https://developer.nvidia.com/cuda-downloads)
- OpenMP developer library
- C++11-aware compiler (eg. GCC toolchain v4.7)

In the project directory you'll find the compile.sh, just run it.

Customization
-------------
To customize the application you can change the following in GameOfLife.cu:
- Board width and height: 69. and 70. line
- Number of steps in the life: 71. line
- Number of streams from which two of them concurrently compute and copy the results back: 72. line
