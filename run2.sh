
cargo test --package libaleominer --lib -- test_synthesis::test1 --nocapture > run.log 
cargo test --package libaleominer --lib -- test_synthesis::test_synthesis_leaves --nocapture > run-my.log   

./comp_circuit_psd2.py ./run.log ./run-my.log

grep "is full" run-my.log

