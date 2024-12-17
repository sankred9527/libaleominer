use std::ffi::c_void;
use std::slice;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaChaRng;
use snarkvm::circuit::{AleoV0, Eject, Plaintext, Value};
use snarkvm::console::account::*;
use snarkvm::ledger::store::helpers::memory::ConsensusMemory;
//use snarkvm::console::network::{MainnetV0, Network};
//use snarkvm::console::types::address::Address;
use snarkvm::ledger::puzzle::{PartialSolution, Puzzle, PuzzleTrait, SolutionID};
use snarkvm::synthesizer::vm::VM;
use ledger_puzzle_epoch::SynthesisPuzzle;
use console_program;
use snarkvm::circuit::integers::Integer;
use snarkvm::circuit::Mode::Constant;
use snarkvm::console;
use snarkvm::prelude::{AleoID, Address, MainnetV0, Network};




fn checked_next_power_of_n(base: usize, n: usize) -> Option<usize> {
    if n <= 1 {
        return None;
    }

    let mut value = 1;
    while value < base {
        value = value.checked_mul(n)?;
    }
    Some(value)
}


#[cfg(test)]
#[test]
fn test_synthesis_leaves()
{
    //let epoch_str = "ab1x0png70w6vj9ejgrtpdzx42w3lkshn67c34xtumhe5ghepg8zs9s3508pw";
    let epoch_str = "ab1rpe2tukk6hd9wrjphdmmk9qlk9cgdnghq98vupl7xu2y6fc5gygs59ezde"; // 70w算力
    //let epoch_str = "ab10jfrtd46rjku9aed09ddxrj5k7ugumucgd8hh2y0tcs8jpul4v8qde74x3"; //
    let epoch_hash = <MainnetV0 as Network>::BlockHash::from_str(&epoch_str).unwrap();

    // let epoch_bytes = [52, 195, 52, 121, 238, 211, 36, 92, 201, 3, 88, 90, 35, 85, 78, 143, 237, 11, 207, 92, 196, 106, 101, 243, 119, 205, 17, 124, 133, 7, 20, 11];
    // let epoch_hash = AleoID::<Field<MainnetV0>, 0x6261>::from_bytes_le(&epoch_bytes).unwrap();

    // let mut rng = rand::thread_rng();
    // let epoch_hash = <MainnetV0 as Network>::BlockHash::rand(&mut rng);

    let puzzle = SynthesisPuzzle::<MainnetV0, AleoV0>::new();
    let epoch_program = puzzle.get_epoch_program(epoch_hash).unwrap();

    let program_bin = crate::convert_epoch_program_to_cuda_struct(&epoch_program);
    crate::miner_testlib_save_program_as_bin( program_bin.as_slice(), program_bin.len() );

    let foo = epoch_hash.to_bytes_le().unwrap();
    //crate::miner_testlib_simple1(foo.as_slice(), program_bin.as_slice(), program_bin.len() );

}

#[test]
fn test1()
{
    let epoch_str =  "ab1eflsr2n5680pa8ce4nrl5f5f6ddlgy0wkkx9ypc8n8x2pvxcgy9s370mmy"; // my fast
    //let epoch_str = "ab17plesf50kms78x7s003j45g9v03ywllz8k2we4zc5tdgqzrajcgsw4ana6";
    let epoch_hash = <MainnetV0 as Network>::BlockHash::from_str(&epoch_str).unwrap();

    let epoch_bytes = [52, 195, 52, 121, 238, 211, 36, 92, 201, 3, 88, 90, 35, 85, 78, 143, 237, 11, 207, 92, 196, 106, 101, 243, 119, 205, 17, 124, 133, 7, 20, 11];
    let epoch_hash = AleoID::<Field<MainnetV0>, 0x6261>::from_bytes_le(&epoch_bytes).unwrap();

    // let mut rng = rand::thread_rng();
    // let epoch_hash = <MainnetV0 as Network>::BlockHash::rand(&mut rng);

    let address_bytes: [u8; 32] =  [76, 64, 99, 46, 103, 207, 243, 204, 83, 127, 160, 168, 217, 1, 175, 68, 100, 181, 49, 225, 155, 172, 149, 85, 110, 217, 46, 178, 245, 249, 87, 10];
    let address = Address::<MainnetV0>::from_bytes_le(&address_bytes).unwrap();
    // let addr_str = "ab1y07ryk6t2lshy5fhlxkykp80wlxn7xxwc3jtj5wf7md0stmu4uysx6fcg5";
    // let address = Address::<MainnetV0>::from_str(addr_str).unwrap();

    let addr_bin = address.to_bytes_le().unwrap();
    println!("addr={:?}", addr_bin);
    println!("epoch={:?}", epoch_hash.to_bytes_le().unwrap());

    let puzzle =
        VM::<MainnetV0, ConsensusMemory<MainnetV0>>::new_puzzle()
            .unwrap();
    let counter = 0x1234567887654321u64;
    //let counter = 0x1234567887658600u64;
    //let counter = 0x1234567887654321u64 + 45;
    let ps = PartialSolution::new(epoch_hash, address, counter).unwrap();

    if false {
        let sid = SolutionID::<MainnetV0>::from_str("solution1rmmqrw78yrrvx7eyfga").unwrap();
        let leaves = puzzle.get_leaves_by_solution_id(&sid, epoch_hash).unwrap();
        // Get the proof target.
        let proof_target = Puzzle::<MainnetV0>::leaves_to_proof_target(&leaves).unwrap();
        println!("solution_target 2={:x}", proof_target);
    } else {
        let leaves = puzzle.get_leaves(&ps).unwrap();
        // Get the proof target.
        let proof_target = Puzzle::<MainnetV0>::leaves_to_proof_target(&leaves).unwrap();
        println!("solution_target 2={:x}", proof_target);
    }
    return ;
}

#[test]
fn test2()
{
    use snarkvm::circuit::types::integers::*;
    use snarkvm::circuit::environment::traits::*;
    use console::network::*;
    type MI8 = snarkvm::circuit::integers::I8::<AleoV0>;


    let n1 = MI8::parse("13i8").unwrap().1;
    let n2 = MI8::parse("13i8").unwrap().1;

    let n3 = n1.mul_wrapped(&n2);

    println!("n3={}", n3);
}
