

file(GLOB_RECURSE CUDA_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"  
)

add_library(aleocuda STATIC ${CUDA_SOURCES} )
set_target_properties(aleocuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON    
)
#target_compile_options(aleocuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
target_link_libraries(aleocuda PRIVATE fmt::fmt )
 