


add_library(testlib_ped64 STATIC 
  "${CMAKE_CURRENT_SOURCE_DIR}/test_ped64.cu"  
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_bigint.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_utils.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_aleo_globals.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/circuit/hash/ped64_base.cu"
)


set_target_properties(testlib_ped64 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON    
)

add_executable(test_ped64 "${CMAKE_CURRENT_SOURCE_DIR}/test_ped64.cpp" )
target_link_libraries(test_ped64 PRIVATE testlib_ped64 )


add_library(testlib_bhp256 STATIC 
  "${CMAKE_CURRENT_SOURCE_DIR}/test_bhp256.cu"  
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_bigint.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_utils.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_aleo_globals.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/circuit/hash/hash_bhp256_bases.cu"
)


set_target_properties(testlib_bhp256 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON    
)

add_executable(test_bhp256 "${CMAKE_CURRENT_SOURCE_DIR}/test_bhp256.cpp" )
target_link_libraries(test_bhp256 PRIVATE testlib_bhp256 )


##########################

add_library(testlib_psd2 STATIC 
  "${CMAKE_CURRENT_SOURCE_DIR}/test_psd2.cu"  
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_bigint.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_utils.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_aleo_globals.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/circuit/hash/poseidon_data.cu"
)


set_target_properties(testlib_psd2 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON    
)

add_executable(test_psd2 "${CMAKE_CURRENT_SOURCE_DIR}/test_psd2.cpp" )
target_link_libraries(test_psd2 PRIVATE testlib_psd2 )


add_library(testlib_keccak STATIC 
  "${CMAKE_CURRENT_SOURCE_DIR}/test_keccak.cu"  
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_bigint.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_aleo_globals.cu"  
)
set_target_properties(testlib_keccak PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON    
)

add_executable(test_keccak "${CMAKE_CURRENT_SOURCE_DIR}/test_keccak.cpp" )
target_link_libraries(test_keccak PRIVATE testlib_keccak )


add_library(testlib_field_to_bigint_with_witeness STATIC 
  "${CMAKE_CURRENT_SOURCE_DIR}/test_field_to_bigint_with_witeness.cu"  
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_bigint.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_aleo_globals.cu"  
)
set_target_properties(testlib_field_to_bigint_with_witeness PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON    
)

add_executable(test_field_to_bigint_with_witeness "${CMAKE_CURRENT_SOURCE_DIR}/test_field_to_bigint_with_witeness.cpp" )
target_link_libraries(test_field_to_bigint_with_witeness PRIVATE testlib_field_to_bigint_with_witeness )


add_library(testlib_curve STATIC 
  "${CMAKE_CURRENT_SOURCE_DIR}/test_curve.cu"  
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_bigint.cu"  
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_utils.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_aleo_globals.cu"  
)
set_target_properties(testlib_curve PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON    
)

add_executable(test_curve "${CMAKE_CURRENT_SOURCE_DIR}/test_curve.cpp" )
target_link_libraries(test_curve PRIVATE testlib_curve )

##############

add_library(testlib_inverse STATIC 
  "${CMAKE_CURRENT_SOURCE_DIR}/test_inverse.cu"  
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_bigint.cu"  
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_utils.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_aleo_globals.cu"  
)
set_target_properties(testlib_inverse PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON    
)

add_executable(test_inverse "${CMAKE_CURRENT_SOURCE_DIR}/test_inverse.cpp" )
target_link_libraries(test_inverse PRIVATE testlib_inverse )

###############

add_library(testlib_poww STATIC 
  "${CMAKE_CURRENT_SOURCE_DIR}/test_poww.cu"  
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_bigint.cu"  
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_utils.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda/cu_aleo_globals.cu"  
)
set_target_properties(testlib_poww PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON    
)

add_executable(test_poww "${CMAKE_CURRENT_SOURCE_DIR}/test_poww.cpp" )
target_link_libraries(test_poww PRIVATE testlib_poww )
