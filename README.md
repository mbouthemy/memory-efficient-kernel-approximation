# Memory Efficient Kernel Approximation
*Authors : Marin BOUTHEMY*


This code is an implementation of the Memory Efficient Kernel Approximation (MEKA) algorithm designed by [Si Si & al.](http://www.jmlr.org/papers/volume18/15-025/15-025.pdf).

To use it just run the main function and test it on the ijcnn1 dataset.

## Requirements
The library has some requirements :
 - Python 3
 - Numpy
 - Pandas

To install all this requirement you can use the requirements.txt.

## Files structure
The library contains the following files:

 - [main.py](https://github.com/Marin35/memory-effficient-kernel-approximation/blob/master/src/main.py) -> Run the algorithm and create differents kernel matrices (based on MEKA, Nystrom and classic computation) and calculate the score for each of the matrix.
 - [meka.py](https://github.com/Marin35/memory-effficient-kernel-approximation/blob/master/src/meka.py) -> Implementation of the MEKA algorithm, composed on the 3 steps.
 - [utils.py](https://github.com/Marin35/memory-effficient-kernel-approximation/blob/master/src/utils.py) -> Functions such as the computation of gaussian kernel or the Nystrom approximation algorithm.

