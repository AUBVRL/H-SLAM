#!/bin/bash 
# gy - 17/7/2019
# Run this script once as it will clean up after itself. Everytime you run it will recompile packages (except opencv)
BuildType="Release"

SCRIPTPATH=$(dirname $0)
if [ $SCRIPTPATH = '.' ]
then
SCRIPTPATH=$(pwd)
fi

mkdir -p $SCRIPTPATH/Thirdparty/CompiledLibs
InstallDir=$SCRIPTPATH/Thirdparty/CompiledLibs

#install system wide dependencies
#================================
sudo apt install libgl1-mesa-dev libglew-dev libsuitesparse-dev libeigen3-dev libboost-all-dev cmake build-essential git libzip-dev freeglut3-dev

#if you have OpenCV3.4 comment out the following and specify the directory later
sudo apt install libjpeg8-dev libpng-dev libtiff5-dev libtiff-dev libavcodec-dev libavformat-dev libv4l-dev libgtk2.0-dev qt5-default v4l-utils
cvVersion=3.4.6
if [ ! -d "$SCRIPTPATH/Thirdparty/opencv-${cvVersion}" ]; then
  DL_opencv="https://github.com/opencv/opencv/archive/${cvVersion}.zip"
  DL_contrib="https://github.com/opencv/opencv_contrib/archive/${cvVersion}.zip"
  cd $SCRIPTPATH/Thirdparty/
  wget -O opencv.zip -nc "${DL_opencv}" && unzip opencv.zip && rm opencv.zip && cd opencv-${cvVersion}
  wget -O opencv_contrib.zip -nc "${DL_contrib}" && unzip opencv_contrib.zip && rm opencv_contrib.zip
fi

cd $SCRIPTPATH/Thirdparty/opencv-${cvVersion} && mkdir -p build && cd build &&cmake .. -DCMAKE_BUILD_TYPE=$BuildType -DCMAKE_INSTALL_PREFIX=$InstallDir -DWITH_V4L=ON -DWITH_CUDA=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_QT=ON -DOPENCV_EXTRA_MODULES_PATH=$SCRIPTPATH/Thirdparty/opencv-${cvVersion}/opencv_contrib-${cvVersion}/modules && make -j $(nproc) && make install && cd .. && rm -r build
#end comment out OpenCV

#Build Thirdparty libs	
#=====================
echo -e "Compiling Pangolin\n"
cd $SCRIPTPATH/Thirdparty/Pangolin
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=$BuildType -DCMAKE_INSTALL_PREFIX=$InstallDir -DDISPLAY_WAYLAND=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF && make -j $(nproc) && make install && cd .. && rm -r build

echo -e "Compiling G2O\n"
cd $SCRIPTPATH/Thirdparty/g2o
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=$BuildType -DCMAKE_INSTALL_PREFIX=$InstallDir #-DG2O_BUILD_APPS=OFF -DG2O_BUILD_EXAMPLES=OFF -DG2O_USE_OPENMP=true
make -j $(nproc) && make install && cd .. && rm -r build && rm -r bin && rm -r lib

echo -e "Compiling DBoW3\n"
cd $SCRIPTPATH/Thirdparty/DBow3
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=$BuildType -DCMAKE_INSTALL_PREFIX=$InstallDir -DUSE_CONTRIB=true -DOpenCV_DIR=$InstallDir/share/OpenCV && make -j $(nproc) && make install && cd .. && rm -r build

#set environment settings
#==========================
if grep -Fxq 'PATH=${PATH}'":${InstallDir}/bin" ~/.bashrc 
then :
else
  echo 'PATH=${PATH}'":${InstallDir}/bin" >> ~/.bashrc 
  echo 'LD_LIBRARY_PATH=${LD_LIBRARY_PATH}'":${InstallDir}/lib" >> ~/.bashrc
  source ~/.bashrc 
fi

#build SLAM
#==========
cmake_prefix=$InstallDir/lib/cmake
cd $SCRIPTPATH && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=$BuildType && make -j


