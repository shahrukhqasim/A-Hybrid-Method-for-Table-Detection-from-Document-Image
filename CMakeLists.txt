cmake_minimum_required(VERSION 3.7)
project(detect)


link_directories(/usr/local/lib/ /usr/lib/ ${Boost_LIBRARY_DIR})
include_directories(/usr/local/include/ csv json/json ${Boost_INCLUDE_DIR})


set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(detect ${SOURCE_FILES})

target_link_libraries(detect lept tesseract opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)