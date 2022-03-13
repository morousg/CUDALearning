# CUDAMemPatterns IP

Most of the kernels I have been working with are memory bound. That means that the number of operations is low compared to the number of memory accesses, due to the nature of the algorithm.

This can happen because the number of operations is vey low, or because the memory accesses are very sparse, and generate a lot of memory access instructions.

Therefore, most of the optimizations that can be done could be summarized in the following:

1. Read and write 128 bit elements per thread, to maximize bandwith and minimize the number of memory instructions (less fetch and decode phases). In turn, this is giving  a greater number of operations to each thread, wich they already can do for free, and improves overall data locality.
2. Fuse kernels when possible in order to read once, do all the operations, and write once. This is probably the most effective optimization, that can make some kernels virtually dissapear, in terms of execution time. So, for the cost of executing one kernel, we may be able to execute 2, 8 or more kernels, depending on the amount of operations on each kernel.
3. Depending on the sparsity of the reads, it can be usefull to use cudaTextures and/or to make the Thread Blocks to iterate over the data, to further improve data spatial locality (avoiding neigboring TB to be scheduled in different SM's).

Looking at those optimizations, we can see that the same optimizations can be applied to many different algorithms. Which means, that we can write a single kernel, with a certain memory pattern implemented in an optimized way, and then pass to it some pixel based operations, that can be executed one after the other. The output of the first operation will be the input of the next. That way, the pixels stay in register memory, instead of being moved to device memory back and forth constantly.

I want to create a basic set kernels, that have no actual operations defined, but do have an speciffic memory access pattern.

The memory patterns I want to work on are:

1. Map operations. Thinking on pixels (or data elements), that would mean to apply some transformations to the values of each pixel.
2. Reduction opertaions. I want a single kernel capable of applying an optimized reduction memory pattern over a matrix, with or without a mask, and be able to apply more than one reduction operation at the same time, including to the mask. That would allow to do the following operations: min, max, sum of all elements, variance or standard deviation, find the value closest to an specific criteria, count the occurrences of a value, etc...
3. Color space transformation. This is a very simple operation, and the optimization is very similar for many cases. Plus, besides implementing a single memory partern that could serve many color space transformations, this kind of kernel is very easy to fuse with other kernels, specially when the output of the transformation is based on Red, Green and Blue, since most of the image based algorithms work in this color spaces.

# Map or Transform pattern:

The implementation of this memory pattern is highly optimized. The actual operations are passed as a template parameter, or as a functor, or using C++ trics and Variadic templates. Specifically, with variadic templates, I want to be able to define an arbitrary number of consecutive operations to be performed on the data, so that memory reads and writes on device memory are reduced to the minimum.

After successfully implementing a cuda transform version, that can apply several consecutive operations with different parameters, with a single kernel definition, by using Variadic Templates, now I want to create a test base, to be able to:

- Compare execution times for different data sizes, of this kernel with N consecutive operations, against executing individual transform kernels, one after the other, as one can usually find in many libraries like thrust and OpenCV. I want to know how much better is this implementation.
- Repeate the same experiment, but this time, use a cuda graph for executing the consecutive transform kernels, to see how much cuda graph can optimize kernel execution. Of course, the individual kernels still will read and write device memory several times, therefore I expect my variadic template transform implementation to still be faster.
- To have a variadic-transform baseline implementation, to compare with possible optimized versions that I will be testing. Also to compare with versions that will be using masks, and ROI's (Regions of Interest).

## Results

Note: dates in european format DD.MM.YYYY

02.01.2019 Achievements:
- A working CUDA kernel that takes any number of binary operations+parameter as parameter, to operate on a vector/matrix. This is the same as std::transform, but without the need for iterators and for any number of perations done consecutively in a single kernel launch, reading and writing data from device memory once.
- The first benchmarks performed with NVIDIA's NSIGHT Visual Sutdio edition, show big speedups. For 5 transform operations, on a 4K sized vector (3840*2160 numbers on a row), with float data type, shows that my implementation is up to 5 times faster.
- The way of using this code, is quite simpler than writting a single kernel for each case.
- An initial attempt of kernel optimization, that is still buggy (probably because of vector type data ordering), shows a 2x speedup compared to the previous optimization.

06.01.2019 Achievements:
- Optimized memory access in transform kernel, improves execution times in about 1,5x for big arrays. So, up to now, the biggest experiment I have done, has changed from 2,750ms to 0,520ms to 0,340ms. Therefore, total speedup is about 8x, for 5 transform operations.

## Conclussions
- Having a 5x speedup for 5 operations makes complete sense. The GPU spends most of it's time reading and writing data. It has plenty of time in between for doing a lot of computations, without making the kernel slower. This is thanks to a very early mechanism on GPU's called latency hiding.
- The conceptual separation of memory patterns and actual operations, makes it more productive to make optimizations on this kernel. Optimizing this single kernel, equates to potentially optimizing hudreds of kernels written without C++ variadic template types. Optimizing the way in which the data is accessed, in the case of transform at least, has nothing to do with the operations being done, and time spent on optimizing this kernel, will benefit all your current and future uses. Also, it makes faster to migrate the code to newer syntaxes of the language like with CUDA 9 and 10.

# Reduce pattern:

## Notes/Ideas

For the reduce kernel, we could think of cases where we need to perform several and different reductions on the same source data. A good example of this is when you want to compute the variance of a matrix or a vector. We need to obtain the sumation of all elements, plus the summation of the square of all the kernels, and the total number of elements in the matrix (or the number of elements used, in case we use a mask or ROI).

In this case the operations of the reduction should be split into three parts:
- The initial operations sum, sum of sqr, and return 2, 1 or 0 according to whether we use both numbers, one or none.
- The rest of the operations, in this case sumation for all cases.
- An optional final operation, in case we want (in this case we want, to compute the variance).

How we pass this information to the kernel? We could use an struct, where we have the two first operations, and a number that represents the index of the device memory pointer where we want this result to be written (we could add a flag to indicate if we operate on the mask or on the source data, a different solution should be implemented when using ROI). We could pass an indefinite number of this structs, using variadic templates. Another struct called something like "nary_operation", would contain an operator that takes the output vector, and operates on the previously generated results, to generate any number of results stored on the same output vector.

Using this strategy, we can do a normal reduction, min, max, compute variance, others, or all of them at the same time, executing the optimized memory pattern, only once.

Quite different strategies should be applied to reduce columns or rows, or to reduce blobs defined by several ROI's.

In general, what should one do with scalar values? This values can be generated on Device memory. Therefore they should be passed as a pointer to the next kernel, to avoid latencies. The operation structs, should some how support the usage of an scalar in a gpu pointer. Probably, the easyest is to create another struct, and overloaded corresponding device functions, operating with it.

# Color space converion
