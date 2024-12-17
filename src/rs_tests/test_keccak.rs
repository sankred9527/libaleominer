

use rand::Rng;
use tiny_keccak::{Hasher, Sha3};

fn test_sha3_rand<const CNT : u32>()
{
    let mut vec: Vec<u8> = vec![0; CNT as usize];    

    let mut rng = rand::thread_rng();
    rng.fill(&mut vec[..]);


    let slice: &[u8] = &vec[..];

    // let mut out1 = [0u8;32];
    // crate::miner_test_keccak(slice, slice.len() as u32 , &mut out1);
    //
    // let mut sha3 = Sha3::v256();
    // let mut out2 = [0u8; 32];
    // sha3.update(slice);
    // sha3.finalize(&mut out2);
    //
    // assert_eq!(out1, out2);
}

#[cfg(test)]
#[test]
fn test_keccak()
{
    test_sha3_rand::<16>();
    test_sha3_rand::<32>();
    test_sha3_rand::<200>();
    test_sha3_rand::<256>();
    test_sha3_rand::<280>();
    test_sha3_rand::<512>();
}