# CUDALearning
This GitHub repository is a place to put all my tests and small projects, done with the purpose of better understanding CUDA and GPU performance behavior, as well as testing ways of improving productivity with pointers (not std::vector or similar) in a C++ environment. I will use C++ for implementing easy interfaces, to use CUDA optimizations. This project does not intend to introduce people into CUDA, although I might put some materials for teaching CUDA.

Also I want to learn and test new CUDA features, and create implementations of GPU-hater algorithms, that are not necessarily complex, but perform poorly on the GPU, and are available for free usually for CPU.

Why do I make it public? Well, I don't want this knowledge to stay in a drawer, it might be useful to someone. And if anyone happens to look at it, he/she might find mistakes or might be able to give me some references of similar work that I don't know of. Or even, maybe solve the questions directly :-).

I will probably use the code in this repo to ask questions in the NVIDIA forums too.

Why not to sign-up in a CUDA course instead? Well, I've started learning CUDA when Compute Capabilites where not even 2.0. Also, OpenCL was not abailable for testing, but I had the specification to learn it. Later I did some courses, and I think that none can answer the questions I want to currently solve. So I preffer to test and share to the community, so if I can't find an answer, hopefully if someone is looking at the same things might be able to help. Of course, what I put here might also be usefull for others. That is why I'm using Unlicense, because I want this code to be used by any one, without restrictions.

The organization of the repository is very simple. A set of folders, where each folder contains a project that serves a purpose.

Each project can be in different status:
- To be defined (TBD): the description of the project is not finished.
- Open (O): the goals of the project are defined and it's ready to start, but it was not yet started.
- In Progress (IP): the execution of the project started but it's not finished.
- Blocked (B): the execution of the project started but it's not finished, and there is something that prevents further development.
- Finished (F): the execution of the project is finished, because I reached the main goals of the project. It may still change if there is something new that I discovered, or if I wanted to do some code cleaning or documentation.

## CUDALearning projects
This section enumerates the different projects, with an small description of each project and the status of the project.

### DeviceToDevice memory copies performance (And philosophy of why to do that?) IP
Project dedicated to the exploration of the best performance possible, when copying data from GPU pointers to GPU pointers. Also, contains discussion about why and how to avoid using this copies.

### New CUDA 9 and 10 features TBD
This project needs not much description. There are some CUDA samples showing how to use the new features in CUDA 9, that I will check. Also, there is one kernel I would like to implement, with the help of global synchronization, which is the efficient summation of uniquely labelled blobs in an image. This has a lot to do with the Reduction algorithm. There are many alternatives for this. The blobs can be values representing blob IDs in an image, or you can use (x,y) coordinates to define bounding boxes arround this blobs, and perform reduction on this area. In the second case, the values can be either just flags 0 or 1, or more interesting values like probabilities, where you may want to do something more than just counting the number of pixels >0. I will describe all the options, and try to implement all of them, ideally with a single kernel, using cuda 9 features and templates. There might be other kernels that might be interesting to implement 

### Separation of CUDA memory patterns and it's actual operations IP
I want to have basic kernels (mainly Map or Transform, Reduce etc...), that have no actual operations defined, but do have an speciffic memory pattern, and the implementation of this memory pattern is highly optimized. The actual operations are passed as a template parameter, or as a functor, or using C++ trics and Variadic templates. Specifically, with variadic templates, I want to be able to define an arbitrary number of consecutive operations to be performed on the data, so that memory reads and writes on device memory are reduced to the minimum.

### cudaMemsetAsync weird behavior in cuda 9.1 F
I found that cudaMemsetAsynch does not respect the fifo nature of CUDA Streams, and also prevents the overlapping of memory transfers and computation, under certain conditions. I want to explore this issues, and possible kernel implementations that might prove to be faster than cudaMemsetAsynch.

### Convolutions and their optimization as matrix multiplication: exploring how to efficiently shape data from RGB images, to the convolutional layer of a DNN. IP

I want to:
- Implement convolution kernels, in the traditional way, to apply basic filters to images, as a basic material for teaching basic cuda, and shared memory usage.
- To review spatial separable convolution kernels, and see improvements with new CUDA features like global synchronization, fp16, what about tensor cores?.
- To review depth wise separable convolutions, and their implementation as matrix multiplication, and the implications in memory movement. Also compare the performance of different implementations: from non matrix multiplication, to matrix multiplication with fp32, fp16, and tensor cores when I have them available.
- To also understand how filters in int8 and int4 can be applied and still get acceptable results.
- Very intresting article by the way: https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728

