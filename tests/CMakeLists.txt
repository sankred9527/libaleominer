


file(GLOB TEST_SOURCES  
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"  

)


add_executable(test_farm "${CMAKE_CURRENT_SOURCE_DIR}/test_farm.cpp"  )
target_link_libraries(test_farm PRIVATE aleocuda )