cmake_minimum_required(VERSION 3.18)
project(Tc1D_native LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

add_library(RDAAM MODULE src/RDAAM_He/RDAAM_He.cpp src/RDAAM_He/RDAAM.cpp)
add_library(ketch MODULE src/ketch_aft/ketch.c)

install(TARGETS RDAAM DESTINATION tc1d)
install(TARGETS ketch DESTINATION tc1d)