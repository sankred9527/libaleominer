
file(GLOB_RECURSE TESTLIB_SOURCES  
  
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cc"
)

add_library(testlib STATIC ${TESTLIB_SOURCES} )

set_target_properties(testlib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON    
)
target_link_libraries(testlib PRIVATE fmt::fmt )

add_executable(test_simple "${CMAKE_CURRENT_SOURCE_DIR}/test_simple.cpp"  )
target_link_libraries(test_simple PRIVATE testlib)

