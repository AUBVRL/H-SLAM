PROJECT(MySLAM)
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

set(BUILD_TYPE Release)
#set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.4.6 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(G2O REQUIRED)
find_package(DBoW3 REQUIRED)
find_package(LibZip QUIET)


if (LIBZIP_LIBRARY)
  add_definitions(-DHAS_ZIPLIB=1)
  include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} )
else()
  set(LIBZIP_LIBRARY "")
endif()

add_definitions("-DENABLE_SSE")

#detect and number of processors
include(ProcessorCount)
ProcessorCount(NUMTHREADS)
add_definitions("-DThreadCount=${NUMTHREADS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -O3 -g -march=native -Wno-deprecated -Wno-deprecated-declarations") # -fopenmp -Wno-deprecated -Wno-deprecated-declarations -fno-omit-frame-pointer"

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)

# Sources files
file(GLOB SOURCES "src/*.cpp")
add_library(SLAMLib ${SOURCES})

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/Include
  ${PROJECT_SOURCE_DIR}/../thirdparty/src/sse2neon
  ${PROJECT_SOURCE_DIR}/../thirdparty/src/Sophus
  ${DBoW3_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
  ${CSPARSE_INCLUDE_DIR}
  ${CHOLMOD_INCLUDE_DIR}
) 

# build executables
set(LIBS
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${G2O_LIBS}
  ${LIBZIP_LIBRARY} 
  ${Pangolin_LIBRARIES}
  ${DBoW3_LIB_DIR}/libDBoW3.so
)

add_executable(MySLAM ${PROJECT_SOURCE_DIR}/Main.cpp )
target_link_libraries(MySLAM SLAMLib ${LIBS} ) 

