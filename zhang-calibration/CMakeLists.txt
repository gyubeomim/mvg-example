cmake_minimum_required(VERSION 3.0)

add_compile_options(-std=c++11)

project(mvg_example)

set(PROJECT_PATH ${PROJECT_SOURCE_DIR})
configure_file(util/util.h.in ../util/util.h)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

if(WIN32)
	set(EIGEN3_INCLUDE_DIRS "c:/Program Files (x86)/Eigen3/include/eigen3/")
endif(WIN32)

include_directories(
  src/
  util/
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  )

add_executable(zhang
  src/zhang_calibration.cc
  )
target_link_libraries(zhang
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  )
