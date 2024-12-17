
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::{ChaChaRng, ChaCha20Core};


#[cfg(test)]
#[test]
fn test_chacha() {    
    struct BoxRngcore
    { 
      pub state:[u8;32] 
    }
  
    impl SeedableRng for BoxRngcore {
      type Seed = [u8; 32];
      #[inline]
      fn from_seed(seed: Self::Seed) -> Self {
        BoxRngcore { state: seed }
      }
    }

    let seed64 : u64 = 1024102410241024;

    //先获取 seed64 对应的 rand::seed_from_u64 的key 值,传给cuda 用
    // let a1 = BoxRngcore::seed_from_u64(seed64);
    // println!("{:?}", a1.state);  
    
    // let key = BoxRngcore::seed_from_u64(seed64).state;
    // let mut out = [0u64;40];
    //
    // crate::miner_test_chacha(&key, &mut out, 40);
    //
    // let mut rng = ChaChaRng::seed_from_u64(seed64);
    // let mut out2 = [0u64;40];
    // for n in 0..40 {
    //     out2[n] = rng.next_u64();
    // }
    //
    // for n in 0..40
    // {
    //     assert_eq!(out[n], out2[n]);
    // }
}