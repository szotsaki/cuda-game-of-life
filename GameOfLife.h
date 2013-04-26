/*
 * GameOfLife.h
 *
 *  Created on: 2013.03.23.
 *      Author: szotsaki
 */

#ifndef GOL_H_
#define GOL_H_

void createRandomCells(bool*, int, int);
void writeResultToFile(const char *fileName, const bool *states, const int width, const int height, const int steps);

#endif /* GOL_H_ */
