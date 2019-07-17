SET(PROJECT_NAME GSLAM)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(BUILD_TYPE Release)
#set(BUILD_TYPE RelWithDebInfo)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread) 
find_package(OpenCV REQUIRED)
find_package(Pangolin 0.2 REQUIRED)

# optional
find_package(LibZip QUIET)

add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "${SSE_FLAGS} -O3 -g -std=c++0x -march=native -fopenmp -Wno-deprecated -Wno-deprecated-declarations") #-fno-omit-frame-pointer"

if (MSVC)
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)

# Sources files
set(uslam_SOURCE
  ${PROJECT_SOURCE_DIR}/src/System.cpp

)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/Include
  ${PROJECT_SOURCE_DIR}/Thirdparty/sse2neon
  ${PROJECT_SOURCE_DIR}/Thirdparty/DBow3
  ${PROJECT_SOURCE_DIR}/Thirdparty/Pangolin/include
  ${EIGEN3_INCLUDE_DIR}
) 


include_directories( ${Pangolin_INCLUDE_DIRS} ) 
set(dso_pangolin_SOURCE_FILES 
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
	  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.cpp
)



# decide if we have openCV
message("--- found OpenCV, compiling dso_opencv library.")
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(dso_opencv_SOURCE_FILES 
	${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
	${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp)


# decide if we have ziplib.
if (LIBZIP_LIBRARY)
	message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
	add_definitions(-DHAS_ZIPLIB=1)
	include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} ) 
else()
	message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
	set(LIBZIP_LIBRARY "")
endif()


# compile main library.
include_directories( ${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}) 
add_library(dso ${uslam_SOURCE} ${dso_opencv_SOURCE_FILES} ${dso_pangolin_SOURCE_FILES})


if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(BOOST_THREAD_LIBRARY boost_thread-mt)
else()
    set(BOOST_THREAD_LIBRARY boost_thread)
endif()

# build main executable (only if we have both OpenCV and Pangolin)

add_executable(dso_dataset ${PROJECT_SOURCE_DIR}/src/main_dso_pangolin.cpp )
target_link_libraries(dso_dataset dso boost_system cxsparse ${BOOST_THREAD_LIBRARY} ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR}/thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o.so)


