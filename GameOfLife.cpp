/*
 * GameOfLife.cpp
 *
 *  Created on: 2013.03.23.
 *      Author: szotsaki
 */

#include <random>
#include <fstream>

#include "GameOfLife.h"

void createRandomCells(bool *state, int width, int height) {
    const int size = width * height;

    std::default_random_engine generator;
    std::bernoulli_distribution distribution(0.15); // More dead cells than living

#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        state[i] = distribution(generator);
    }
}

void writeResultToFile(const char* fileName, const bool* states, const int width, const int height, const int steps) {
    std::ofstream output;
    output.open(fileName, std::ios::trunc);

    if (output.is_open()) {
        output << "width = " << width << "\n";
        output << "height = " << height << "\n";
        output << "frames = " << steps << "\n";
        output << "speed = 1000\n";

        int offset = 0;
        for (int i = 0; i < steps; ++i) {
            output << "\n";
            output << "#State no. " << (i + 1) << "\n";

            offset = width * height * i;
            for (int j = 0; j < width; ++j) {
                for (int k = 0; k < width; ++k) {
                    output << (states[offset + height * j + k] ? "*" : ".");
                }
                output << "\n";
            }
        }

        output.close();
    }
}

