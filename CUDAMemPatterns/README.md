# CUDAMemPatterns: 
Let's try to abstract memory patterns from actual operations, and merge consecutive kernel calls.

I want to have basic kernels (mainly Map or Transform, Reduce etc...), that have no actual operations defined, but do have an speciffic memory pattern, and the implementation of this memory pattern is highly optimized. The actual operations are passed as a template parameter, or as a functor, or using C++ trics and Variadic templates. Specifically, with variadic templates, I want to be able to define an arbitrary number of consecutive operations to be performed on the data, so that memory reads and writes on device memory are reduced to the minimum.

After successfully implementing a cuda transform version, that can apply several consecutive operations with different parameters, with a single kernel definition, by using Variadic Templates, now I want to create a test base, to be able to:

- Compare execution times for different data sizes, of this kernel with N consecutive operations, against executing individual transform kernels, one after the other, as one can usually find in many libraries like thrust and OpenCV. I want to know how much better is this implementation.
- Repeate the same experiment, but this time, use a cuda graph for executing the consecutive transform kernels, to see how much cuda graph can optimize kernel execution. Of course, the individual kernels still will read and write device memory several times, therefore I expect my variadic template transform implementation to still be faster.
- To have a variadic-transform baseline implementation, to compare with possible optimized versions that I will be testing. Also to compare with versions that will be using masks, and ROI's (Regions of Interest).
