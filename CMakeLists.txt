cmake_minimum_required(VERSION 3.22)
project("aleo-cuda" LANGUAGES CUDA CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

# 设置编译器标志以禁用优化并启用调试信息
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress=2464 --diag-suppress=177 --ptxas-options=-v -Xptxas -O3, -v " )
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount 255")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas --register-usage-level=10")


if(DEFINED ALEO_DEBUG)
  add_compile_definitions(ALEO_DEBUG=1)
endif()

if(DEFINED DUMP_LEAF)
  add_compile_definitions(DUMP_LEAF=1)
endif()


add_compile_definitions(DEV_BUILD=0)

# 默认设置构建类型
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/libdevcore/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/libaleocore/")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}/src/")


add_subdirectory(fmt)
if(DEFINED EPOCH_PROGRAM_TEST)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
  add_subdirectory(program_test)
elseif(DEFINED TEST_LIB)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
  add_subdirectory(test_lib)
else()  
  add_subdirectory(src)
  add_subdirectory(tests)
endif()


