cmake_minimum_required (VERSION 3.11.0)

include(cmake/parsearguments.cmake)
include(cmake/virtualfolders.cmake)

####################################################################################################
# macro to parse macro named arguments
####################################################################################################
include(cmake/parsearguments.cmake)
include(cmake/virtualfolders.cmake)

###############################
#DeviceToDevice project
###############################
#name of the project 
project(CudaLearning LANGUAGES CXX CUDA)
#include (cmake/libs/cuda.cmake)

#set common include dirs and link directories
include_directories("${CMAKE_SOURCE_DIR}")
 add_subdirectory(src)
#Organize projects into virtual folders
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# avoid minRelSize/RelWithDebugInfo configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)
    
   