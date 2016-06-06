# cis4930project1
CIS4930 Programming Massively Parallel Systems - Project 1

# Problem Statement
This project is about computing spatial distance histogram (SDH) of a collection of 3D points. The SDH problem can be formally described as follows: given the coordinates of N particles (e.g., atoms, stars, moving objects in different applications) and a user-defined distance w, we need to compute the number of particle-to-particle distances falling into a series of ranges (named buckets) of width w: [0, w), [w, 2w), . . . , [(l − 1)w, lw]. Essentially, the SDH provides an ordered list of non-negative integers H = (h0, h1, . . . , hl−1), where each hi(0 ≤ i < l) is the number of distances falling into the bucket [iw, (i + 1)w). Clearly, the bucket width w is a key parameter of an SDH to be computed.

# Getting started and compiling
For you to get started, we have written a C program to compute SDH. The attached file SDH.cu shows a sample program for CPUs and serves as the starting point of your coding in this project. Interesting thing is that you can still compile and run it under the CUDA environment. Specifically, you can type the following command to compile it.
```
      nvcc SDH.cu -o SDH
```
      
To run the code, you add the following line to your testing script file:
```
      ./SDH 10000 500.0
```
Note that the executable takes two arguments, the first one is the total number of data points and the second one is the bucket width (w) in the histogram to be computed. We strongly suggest you compile and run this piece of code before you start coding.


# Tasks to Perform
You have to write a CUDA program to implement the same functionality as shown in the CPU program. Both CUDA kernel function and CPU function results should be displayed as output.
Write a CUDA kernel that computes the distance counts in all buckets of the SDH, by making each thread work on one input data point. After all the points are processed, the resulting SDH should be copied back to a host data structure.
1. Transfer the input data array (i.e., atom list as shown in the sample code) onto GPU device using CUDA functions.
2. Write a kernel function to compute the distance between one point to all other points and update the histogram accordingly. Note that between any two points there should be only one distance counted.
 1
3. Copy the final histogram from the GPU back to the host side and output them in the same format as we did for the CPU results.
4. Compare this histogram with the one you computed using CPU bucket by bucket. Output any differences between them - this should be done by printing out a histogram with the same number of buckets as the original ones except each bucket contains the difference in counts of the corresponding buckets.
Note that the output of your program should only contain the following: (1) the two histograms, one computed by CPU only and the other computed by your GPU kernel. (2) any difference you found between these two histograms.
