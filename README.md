# CUDALearning
This GitHub repository is a place to put all my tests and small projects, done with the purpose of better understanding CUDA and GPU performance behavior, as well as testing ways of improving productivity. I use pointers (not std::vector or similar) in a C++ environment. I will use C++ for implementing easy interfaces, to use CUDA optimizations. This project does not intend to introduce people into CUDA, although I might put some materials for teaching CUDA.

Also I want to learn and test new CUDA features, and create implementations of GPU-hater algorithms, that are not necessarily complex, but perform poorly on the GPU, and are available for free usually for CPU.

Why do I make it public? Well, I don't want this knowledge to stay in a drawer, it might be useful to someone. And if anyone happens to look at it, he/she might find mistakes or might be able to give me some references of similar work that I don't know of. Or even, maybe solve the questions directly :-).

I will probably use the code in this repo to ask questions in the NVIDIA forums too.

Why not to sign-up in a CUDA course instead? Well, I did some courses, and I think that none can answer the questions I want to currently solve. So I preffer to test and share to the community. I'm using Unlicense, because I want this code to be used by any one, without restrictions.

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

### Separation of CUDA memory patterns and it's actual operations IP
I want to have basic kernels (mainly Map or Transform, Reduce etc...), that have no actual operations defined, but do have an speciffic memory pattern, and the implementation of this memory pattern is highly optimized. The actual operations are passed as a template parameter, or as a functor, or using C++ trics and Variadic templates. Specifically, with variadic templates, I want to be able to define an arbitrary number of consecutive operations to be performed on the data, so that memory reads and writes on device memory are reduced to the minimum.

### cudaMemsetAsync weird behavior in cuda 9.1 F
I found that cudaMemsetAsynch does not respect the fifo nature of CUDA Streams, and also prevents the overlapping of memory transfers and computation, under certain conditions. I want to explore this issues, and possible kernel implementations that might prove to be faster than cudaMemsetAsynch.

