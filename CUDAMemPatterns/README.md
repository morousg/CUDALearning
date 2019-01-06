# CUDAMemPatterns IP 
Let's try to abstract memory patterns from actual operations, and merge consecutive kernel calls.

I want to have basic kernels (mainly Map or Transform, Reduce etc...), that have no actual operations defined, but do have an speciffic memory pattern, and the implementation of this memory pattern is highly optimized. The actual operations are passed as a template parameter, or as a functor, or using C++ trics and Variadic templates. Specifically, with variadic templates, I want to be able to define an arbitrary number of consecutive operations to be performed on the data, so that memory reads and writes on device memory are reduced to the minimum.

After successfully implementing a cuda transform version, that can apply several consecutive operations with different parameters, with a single kernel definition, by using Variadic Templates, now I want to create a test base, to be able to:

- Compare execution times for different data sizes, of this kernel with N consecutive operations, against executing individual transform kernels, one after the other, as one can usually find in many libraries like thrust and OpenCV. I want to know how much better is this implementation.
- Repeate the same experiment, but this time, use a cuda graph for executing the consecutive transform kernels, to see how much cuda graph can optimize kernel execution. Of course, the individual kernels still will read and write device memory several times, therefore I expect my variadic template transform implementation to still be faster.
- To have a variadic-transform baseline implementation, to compare with possible optimized versions that I will be testing. Also to compare with versions that will be using masks, and ROI's (Regions of Interest).

## Notes/Ideas

For the reduce kernel, we could think of cases where we need to perform several and different reductions on the same source data. A good example of this is when you want to compute the variance of a matrix or a vector. We need to obtain the sumation of all elements, plus the summation of the square of all the kernels, and the total number of elements in the matrix (or the number of elements used, in case we use a mask or ROI).

In this case the operations of the reduction should be split into three parts:
- The initial operations sum, sum of sqr, and return 2, 1 or 0 according to whether we use both numbers, one or none.
- The rest of the operations, in this case sumation for all cases.
- An optional final operation, in case we want (in this case we want, to compute the variance).

How we pass this information to the kernel? We could use an struct, where we have the two first operations, and a number that represents the index of the device memory pointer where we want this result to be written. We could pass an indefinite number of this structs, using variadic templates. Another struct called somthing like "nary_operation", would contain an operator that takes the output vector, and operates on the previously generated results, to generate any number of results stored on the same output vector.

## Results

Note: dates in european format DD.MM.YYYY

02.01.2019 Achievements:
- A working CUDA kernel that takes any number of binary operations+parameter as parameter, to operate on a vector/matrix. This is the same as std::transform, but without the need for iterators and for any number of perations done consecutively in a single kernel launch, reading and writing data from device memory once.
- The first benchmarks performed with NVIDIA's NSIGHT Visual Sutdio edition, show big speedups. For 5 transform operations, on a 4K sized vector (3840*2160 numbers on a row), with float data type, shows that my implementation is up to 5 times faster.
- The way of using this code, is quite simpler than writting a single kernel for each case.
- An initial attempt of kernel optimization, that is still buggy (probably because of vector type data ordering), shows a 2x speedup compared to the previous optimization.

## Conclussions
- Having a 5x speedup for 5 operations makes complete sense. The GPU spends most of it's time reading and writing data. It has plenty of time in between for doing a lot of computations, without making the kernel slower. This is thanks to a very early mechanism on GPU's called latency hiding.
- The conceptual separation of memory patterns and actual operations, makes it more productive to make optimizations on this kernel. Optimizing this single kernel, equates to optimizing hudreds of kernels written without C++ variadic template types. Optimizing the way in which the data is accessed, in the case of transform at least, has nothing to do with the operations being done, and time spent on optimizing this kernel, will benefit all your current and future uses. Also, it makes faster to migrate the code to newer syntaxes of the language like with CUDA 9 and 10.
