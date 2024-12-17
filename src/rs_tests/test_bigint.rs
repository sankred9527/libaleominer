use snarkvm::utilities::{BigInteger, BigInteger256};

/// Calculate a + (b * c) + carry, returning the least significant digit
/// and setting carry to the most significant digit.
#[inline(always)]
fn mac_with_carry(a: u64, b: u64, c: u64, carry: &mut u64)  {
    let tmp = (u128::from(a)) + u128::from(b) * u128::from(c) + u128::from(*carry);
    
    let t1 = u128::from(b) * u128::from(c);
    println!("b*c={:x}", t1);

    let t2 = (u128::from(a)) + t1;
    println!("a+b*c={:x}", t2);

    let t3 = t2 + u128::from(*carry);
    println!("end={:x}", t3);

    *carry = (tmp >> 64) as u64;

    println!("ret={:x}, {:x}", tmp as u64, (tmp >> 64) as u64);
}


#[test]
fn test_mac_with_carry()
{
    let mut a = 0x1;
    let b = 0x8765432112345678;
    let c = 0xA5559999BBCCCCCC;
    let mut carry = 0x120127812u64;
    
    // mac_with_carry(a, b, c, &mut carry);
    // println!("carry={:x}", carry);

    a = 0xfffffffffffffff1;
    carry = 0x120127812u64;
    mac_with_carry(a, b, c, &mut carry);
    println!("carry={:x}", carry);
}

#[test]
fn test_int256_div2()
{
    let mut v = 0x1234567887654321u64;
    let mut n1 = BigInteger256([v, v+1, v+2, v+3]);

    let mut v = 0x8765432ff2345678u64;
    let i = 0xffff5fffu64;
    let mut n2 = BigInteger256([v, v+i*1, v+i*2, v+i*3]);

    // let mut n1 = BigInteger256([v, 0, 0, 0]);
    // let mut n2 = BigInteger256([v+1, 0,0, 0]);

    //n1.div2();
    n1.sub_noborrow(&n2);
    // use core::arch::x86_64::_subborrow_u64;
    //
    // let mut n1 = 101u64;
    // let mut n2 = 101u64;
    // let mut ret : u64 = 0u64;
    // unsafe {
    //     let mut borrow = 1;
    //     let borrow = _subborrow_u64(borrow, n1, n2, &mut ret);
    //     println!("borrow={}, ret={:x}", borrow, ret);
    // }

    for v in n1.0 {
        println!("n={:x}", v );
    }
}