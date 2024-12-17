
use sha2::{Digest, Sha256};
use std::convert::TryInto;
extern crate snarkvm;

use snarkvm::{console::{
    account::*,
    network::{AleoID, MainnetV0, Network},
}, ledger::{puzzle::PartialSolution, solutions}, synthesizer::snark::ProvingKey};

use snarkvm::ledger::puzzle::{Puzzle,SolutionID, Solution};
//use snarkvm_ledger_puzzle_epoch::MerklePuzzle;        
use rand::{self, CryptoRng, RngCore};

//use crate::miner_test_solution_new;

fn double_sha256(data: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(Sha256::digest(data));
    let mut ret = [0u8; 32];
    ret.copy_from_slice(&digest);
    ret
}

fn sha256d_to_u64(data: &[u8]) -> u64 {
    let hash_slice = double_sha256(data);
    u64::from_le_bytes(hash_slice[0..8].try_into().expect("data to u64 failed"))           
}

#[cfg(test)]
#[test]
fn test_solution()
{
    

    // let rng = &mut rand::thread_rng();

    // // // // Initialize a new puzzle.
    // let puzzle = Puzzle::<MainnetV0>::new::<MerklePuzzle<MainnetV0>>();

    // // // Initialize an epoch hash.
    // let mut epoch_bits = [51, 195, 52, 121, 238, 211, 36, 92, 201, 3, 88, 90, 35, 85, 78, 143, 237, 11, 207, 94, 196, 106, 101, 243, 119, 205, 17, 124, 133, 7, 20, 11];
    // //let epoch_hash : AleoID::<Field<MainnetV0>, 0x6261>= rng.gen();
    // let epoch_hash = AleoID::<Field<MainnetV0>, 0x6261>::from_bytes_le(&epoch_bits).unwrap();
    
    // //let private_key : PrivateKey<MainnetV0> = PrivateKey::new(rng).unwrap();
    // let foo : [u8;32]= [153, 100, 186, 204, 11, 105, 232, 129, 24, 185, 216, 169, 191, 183, 9, 93, 125, 103, 122, 14, 208, 39, 9, 240, 193, 150, 237, 96, 78, 207, 203, 3];    
    // let address = Address::<MainnetV0>::from_bytes_le(&foo).unwrap();
    
    // //let address: Address<MainnetV0> = Address::from_bytes_le(&address_bits).unwrap();
    // let counter : u64 = 0x1234567887654321;    
    
    // println!("epoch-hash={:?}", epoch_hash.to_bytes_le().unwrap());
    // println!("address={:?}, {}", address.to_bytes_le().unwrap(), address.to_bytes_le().unwrap().len());

    // /*
    // let solution = Solution::new(epoch_hash, address, counter)?;
    // 和cuda 实现的 Solution::new 结果一致
    //  */
    // //let solution = Solution::new(epoch_hash, address, counter).unwrap()
    // let sid = SolutionID::new(epoch_hash, address, counter).unwrap();
    // println!("solution id={:?}", sid.to_bytes_le().unwrap());

    // let ret1 = u64::from_le_bytes(sid.to_bytes_le().unwrap().as_slice()[0..8].try_into().expect("data to u64 failed"));
    // println!("ret1={ret1}");

    // let mut output : [u8;8] = [0u8;8];
    // let binding = epoch_hash.to_bytes_le().unwrap();
    // let e = binding.as_slice();
    // let binding = address.to_bytes_le().unwrap();
    // let a = binding.as_slice();
    // //println!("{},{}", e.len(), a.len());
    // miner_test_solution_new(e, a, counter, &mut output);
    // println!("cuda solution id={:?}", output);

    // let ret2 = u64::from_le_bytes(output.try_into().expect("data to u64 failed"));
    // println!("ret2={ret2}");
    
    // assert_eq!(ret1,ret2);
}

#[test]
fn test_num_leaves()
{
    // let puzzle = Puzzle::<MainnetV0>::new::<MerklePuzzle<MainnetV0>>();

    // let mut epoch_bits = [51, 195, 52, 121, 238, 211, 36, 92, 201, 3, 88, 90, 35, 85, 78, 143, 237, 11, 207, 94, 196, 106, 101, 243, 119, 205, 17, 124, 133, 7, 20, 11];
    // //let epoch_hash : AleoID::<Field<MainnetV0>, 0x6261>= rng.gen();
    // let epoch_hash = AleoID::<Field<MainnetV0>, 0x6261>::from_bytes_le(&epoch_bits).unwrap();    

    // let foo : [u8;32]= [153, 100, 186, 204, 11, 105, 232, 129, 24, 185, 216, 169, 191, 183, 9, 93, 125, 103, 122, 14, 208, 39, 9, 240, 193, 150, 237, 96, 78, 207, 203, 3];    
    // let address = Address::<MainnetV0>::from_bytes_le(&foo).unwrap();
    // let counter = 0x1234567887654321;
    // let psolution = PartialSolution::new(epoch_hash, address, counter).unwrap();    
    // println!("solution id={:x}", u64::from_le_bytes(psolution.id().to_bytes_le().unwrap().as_slice().try_into().unwrap()) );
    
    // let solution = Solution::new(psolution, 100);
    // //let seed = u64::from_bytes_le(&epoch_hash.to_bytes_le().unwrap()[0..8]).unwrap();

    // //模拟 snarkVM:  get_proof_target() 函数
    // //第一步：  puzzle.get_leaves()
    // let leaves = puzzle.get_leaves(&psolution).unwrap();

    // println!("leaves len={}", leaves.len());

}