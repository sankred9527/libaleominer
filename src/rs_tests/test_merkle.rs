





#[cfg(test)]
#[test]
fn test_merkle()
{
    use rand_chacha::rand_core::le;
    // use snarkvm::{console::{
    //     account::*,
    //     network::{AleoID, MainnetV0, Network},
    // }, file::ProverFile, ledger::puzzle::{PartialSolution, SolutionID}, synthesizer::snark::ProvingKey};
    //use snarkvm::console::account::*;
    use snarkvm::{
        
        ledger::puzzle::PartialSolution,
        prelude::{
            store::helpers::memory::ConsensusMemory, Address, MainnetV0, Network,
            ToBytes, Uniform, VM,
        },
    };
    
    //use snarkvm::ledger::puzzle::{Puzzle, PuzzleSolutions, Solution};
    //use snarkvm_ledger_puzzle_epoch::MerklePuzzle;

    //let puzzle = Puzzle::<MainnetV0>::new::<MerklePuzzle<MainnetV0>>();
    let puzzle = VM::<MainnetV0, ConsensusMemory<MainnetV0>>::new_puzzle()
                        .unwrap();

    // let epoch_bytes = [52, 195, 52, 121, 238, 211, 36, 92, 201, 3, 88, 90, 35, 85, 78, 143, 237, 11, 207, 92, 196, 106, 101, 243, 119, 205, 17, 124, 133, 7, 20, 11];
    // let epoch_hash = AleoID::<Field<MainnetV0>, 0x6261>::from_bytes_le(&epoch_bytes).unwrap();
    
    // let epoch_str = "ab1x0png70w6vj9ejgrtpdzx42w3lkshn67c34xtumhe5ghepg8zs9s3508pw";
    // let epoch_hash = <MainnetV0 as Network>::BlockHash::from_str(&epoch_str).unwrap();

    // let address_bytes: [u8; 32] =  [76, 64, 99, 46, 103, 207, 243, 204, 83, 127, 160, 168, 217, 1, 175, 68, 100, 181, 49, 225, 155, 172, 149, 85, 110, 217, 46, 178, 245, 249, 87, 10];
    // let address = Address::<MainnetV0>::from_bytes_le(&address_bytes).unwrap();

    let mut rng = rand::thread_rng();
    let epoch_hash = <MainnetV0 as Network>::BlockHash::rand(&mut rng);
    let address = Address::<MainnetV0>::rand(&mut rng);

    let counter : u64 = 0x1234567887655eee;
    // //let counter : u64 = 0x1234567887654321;
    // println!("epoch-hash={:?}", epoch_hash.to_bytes_le().unwrap());
    // println!("address={:?}", address.to_bytes_le().unwrap());

    for rc in  counter..counter+1
    {
        let solution =  PartialSolution::new(epoch_hash, address, rc).unwrap();  
        let proof_target = puzzle.get_proof_target_from_partial_solution(&solution).unwrap();
        //if proof_target > 100 {
            let sid = u64::from_le_bytes(solution.id().to_bytes_le().unwrap().as_slice().try_into().unwrap());
            println!("sid={:x}, counter={:x}, proof target={:x}", sid, rc, proof_target);      
        //}
    }
    
    

    // 模拟 snarkVM 里的：  ledger/puzzle/src/lib.rs  prove() 函数
    // let solution = Solution::new(epoch_hash, address, counter).unwrap();    
    // println!("solution id={:x}", u64::from_le_bytes(solution.id().to_bytes_le().unwrap().as_slice().try_into().unwrap()) );
    
    //let seed = u64::from_bytes_le(&epoch_hash.to_bytes_le().unwrap()[0..8]).unwrap();

    //模拟 snarkVM:  get_proof_target() 函数
    //第一步：  puzzle.get_leaves()
    //let leaves = puzzle.get_leaves(&solution).unwrap();

    

    /*
    Field<Param> 的随机初始化，看 macro impl_primefield_standard_sample
    Field的Param 是 snarkvm_fields::fp_256::Fp256<snarkvm_curves::bls12_377::fr::FrParameters 
    Fp256 的底层是 BigInteger256 ，  先调用 BigInteger256(rng.gen()) ,    
    生成 Filed<Param> 的参数定义在 curves/src/bls12_377/fr.rs ：
    const MODULUS: BigInteger = BigInteger([
        725501752471715841u64,
        6461107452199829505u64,
        6968279316240510977u64,
        1345280370688173398u64,
    ]);

    Field<Fp256>.to_bits_le的时候，要调用 Fp256::to_bigint() 把随机的256bit 整数进行转化
     */

    

    //println!("leaves={:?}", leaves);
    // let total : usize = leaves.iter().map(|inner_vec| inner_vec.len()).sum();
    // println!("total={total}");

    // let t = puzzle.get_proof_target(&sol).unwrap();
    //let proof_target = Puzzle::leaves_to_proof_target(&leaves).unwrap();

    println!("end");
}