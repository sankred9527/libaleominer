

export CUDAARCHS='75;80;86;89'

export CUDAARCHS='89'

# build ffi for rust 
# cmake .. -DTEST_LIB=1 

# build unit test 
# cmake .. -DEPOCH_PROGRAM_TEST=1 

# just build the libarary
cmake ..

