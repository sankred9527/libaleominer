# libaleominer

This is aleo ( https://aleo.org/  ) miner library , it implements snark circuit R1CS and merkle tree in cuda .

If wanna run real miner with pool service , you can wrap it with golang,rust or any other languages. 


# compare result

对比 cuda 和 snarkVM 官方代码的结果是否一致 , compare the result of CUDA  with snarkVM cpu 

cargo test --package libaleominer --lib -- test_synthesis::test_gpu --nocapture | tee run-gpu.log   
cargo test --package libaleominer --lib -- test_synthesis::test_cpu --nocapture | tee run-cpu.log 

./comp_circuit.py ./run-cpu.log ./run-gpu.log


# how to build

read the build_all_cuda_arch.sh 

# how to use 

read example in tests/test_farm.cpp  for how to use it . 