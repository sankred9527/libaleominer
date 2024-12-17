

#[cfg(test)]
#[test]
fn test_sha256()
{
    use sha2::{Digest, Sha256};
    use std::convert::TryInto;
    
    pub fn double_sha256(data: &[u8]) -> [u8; 32] {
        let digest = Sha256::digest(Sha256::digest(data));
        let mut ret = [0u8; 32];
        ret.copy_from_slice(&digest);        
        ret
    }
    
    pub fn sha256d_to_u64(data: &[u8]) -> u64 {
        let hash_slice = double_sha256(data);
        u64::from_le_bytes(hash_slice[0..8].try_into().expect("data to u64 failed"))           
    }
        
    let data : [u8; 8] = [ 0,1,2,3,4,5,6,7 ];
    let r1 = sha256d_to_u64(&data);

    // let mut output_data  = [0u8; 32];
    // //crate::miner_run_sha256_double(&data, 8 , &mut output_data);
    // crate::miner_test_for_sha256d(&data, 8 , &mut output_data);
    // let r2 = u64::from_le_bytes(output_data[0..8].try_into().expect("data to u64 failed"));
    //
    // println!("{r1},{r2}");
    // assert_eq!(r1,r2);
}