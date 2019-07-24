SET(PROJECT_NAME FSLAM)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(BUILD_TYPE Release)
#set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# C++ standard
if(${CMAKE_VERSION} VERSION_LESS "3.2.0")
  set( CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}" )
else()
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(Boost COMPONENTS system thread) 
find_package(OpenCV 3.4.6 REQUIRED PATHS ${PROJECT_SOURCE_DIR}/Thirdparty/CompiledLibs/share/OpenCV)
find_package(Pangolin REQUIRED PATHS ${PROJECT_SOURCE_DIR}/Thirdparty/CompiledLibs/lib/cmake/Pangolin)
find_package(G2O REQUIRED)
find_package(DBoW3 REQUIRED PATHS ${PROJECT_SOURCE_DIR}/Thirdparty/CompiledLibs/lib/cmake/DBoW3)
find_package(LibZip QUIET)

if (LIBZIP_LIBRARY)
  add_definitions(-DHAS_ZIPLIB=1)
  include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} )
else()
  set(LIBZIP_LIBRARY "")
endif()

add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -O3 -g -march=native -fopenmp -Wno-deprecated -Wno-deprecated-declarations ") #-fno-omit-frame-pointer"

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)

# Sources files
set(fslam_SOURCE
  ${PROJECT_SOURCE_DIR}/Src/System.cpp
  ${PROJECT_SOURCE_DIR}/Src/Settings.cpp
  ${PROJECT_SOURCE_DIR}/Src/GeometricUndistorter.cpp

)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/Include
  ${PROJECT_SOURCE_DIR}/Thirdparty/sse2neon
  ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
  ${DBoW3_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
  ${CSPARSE_INCLUDE_DIR}
  ${CHOLMOD_INCLUDE_DIR}
) 

# compile library.
add_library(FSLAMlib ${fslam_SOURCE})

#if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#    set(BOOST_THREAD_LIBRARY boost_thread-mt)
#else()
#    set(BOOST_THREAD_LIBRARY boost_thread)
#endif()

# build executables
set(LIBS
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${G2O_LIBS}
  ${LIBZIP_LIBRARY} 
  ${Pangolin_LIBRARIES}
  ${DBoW3_LIB_DIR}/libDBoW3.so
) #${BOOST_THREAD_LIBRARY} 
add_executable(FSLAM ${PROJECT_SOURCE_DIR}/Main.cpp )
target_link_libraries(FSLAM FSLAMlib cxsparse ${LIBS} ) #boost_system


