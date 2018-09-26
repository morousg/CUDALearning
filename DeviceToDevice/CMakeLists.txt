
###############################
#DeviceToDevice project
###############################
#name of the project 
#we store in variable SRCS all .cpp, .h and .rc files
file(GLOB_RECURSE SRCS *.cpp *.h *.cu )
#set common include dirs and link directories
include_directories("${CMAKE_SOURCE_DIR}")
add_executable(DeviceToDevice  WIN32 ${SRCS})
set_target_properties(DeviceToDevice PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
 
#Organize projects into virtual folders
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)


    
   