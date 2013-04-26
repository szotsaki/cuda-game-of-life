g++ -c GameOfLife.cpp -std=c++11 -fPIC -fopenmp -o GameOfLife.o -Wall -Wextra
nvcc -Xcompiler -fPIC -Xcompiler -Wall -Xcompiler -Wextra -c GameOfLife.cu -o GameOfLife.co
g++ -o GameOfLife *.o *.co -lcudart -fopenmp -L /usr/local/cuda/lib64/
