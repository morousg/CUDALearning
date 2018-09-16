# CUDALearning
This GitHub repository is a place to put all my tests/projects, done with the purpose of better understanding CUDA and GPU performance behavior. This project does not intend to introduce people into CUDA, although I might put some materials for teaching CUDA.

Also I want to learn and test new CUDA features, and create implementations of GPU-hater algorithms, that are not necessarily complex, but perform poorly on the GPU, and are available for free usually for CPU.

Why do I make it public? Well, I don't want this knowledge to stay in a drawer. And if anyone happens to look at it, he/she might find mistakes or might be able to give me some references or similar work that I don't know of. Or even, maybe solve the questions directly :-).

I will probably use the code in this repo to ask questions in the NVIDIA forums too.

Why not to sign-up in a CUDA course instead? Well, I've started learning CUDA when Compute Capabilites where not even 2.0. Also, OpenCL was not abailable for testing, but I had the specification to learn it. Later I did some courses, and I think that none can answer the questions I want to currently solve. So I preffer to test and share to the community, so if I can't find an answer, hopefully if someone is looking at the same things might be able to help. Of course, what I put here might also be usefull for others. That is why I'm using Unlicense, because I want this code to be used by any one, without restrictions.

The organization of the repository is very simple. A set of folders, where each folder contains a project that serves a purpose.

Each project can be in different status:
- To be defined (TBD): the description of the project is not finished.
- Open: the goals of the project are defined and it's ready to start, but it was not yet started.
- In Progress: the execution of the project started but it's not finished.
- Blocked: the execution of the project started but it's not finished, and there is something that prevents further development.
- Finished: the execution of the project is finished, because I reached the main goals of the project. It may still change if there is something new that I discovered, or if I wanted to do some code cleaning or documentation.

## CUDALearning projects
This section enumerates the different projects, with an small description of each project and the status of the project.

### DeviceToDevice memory copies performance (And philosophy of why to do that?)
Project dedicated to the exploration of the best performance possible, when copying data from GPU pointers to GPU pointers. Also, contains discussion about why and how to avoid using this copies.

### New CUDA 9 features TBD
This project needs not much description. There are some CUDA samples showing how to use the new features in CUDA 9, that I will check. Also, there is one kernel I would like to implement, with the help of global synchronization, which is the efficient summation of uniquely labelled blobs in an image. This has a lot to do with the Reduction algorithm. There are many alternatives for this. The blobs can be values representing blob IDs in an image, or you can use (x,y) coordinates to define bounding boxes arround this blobs, and perform reduction on this area. In the second case, the values can be either just flags 0 or 1, or more interesting values like probabilities, where you may want to do something more than just counting the number of pixels >0. I will describe all the options, and try to implement all of them, ideally with a single kernel, using cuda 9 features and templates. There might be other kernels that might be interesting to implement 

### Separation of CUDA memory patterns and it's actual operations TBD
I want to have basic kernels (mainly Map or Transform, Reduce etc...), that have no actual operations defined, but do have an speciffic memory pattern, and the the implementation of this memory pattern is highly optimized. Not only that, but I want to be able to define an arbitrary number of consecutive operations to be performed
