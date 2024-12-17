
use core::ffi::c_void;
use ledger_puzzle_epoch::EpochProgram;
use snarkvm::{ledger::puzzle::SolutionID, prelude::Network};
use snarkvm::synthesizer::program::ProgramCore;
// use core::ffi::c_uchar;
// use std::process::Output;
// use std::ptr::NonNull;


mod rs_tests;
mod cuinterface;

extern "C" {
    fn cuminer_getNumDevices() -> core::ffi::c_int;
    fn cuminer_enumDevices() -> c_void;


    fn testlib_save_program_as_bin(epoch_program : *const u8, program_size : usize);

    fn testlib_simple1(epoch_bin : *const u8, epoch_program : *const u8, program_size : usize);

}

pub fn miner_testlib_simple1(epoch_bin : &[u8], epoch_program: &[u8], program_size: usize)
{
    unsafe  {
        testlib_simple1(epoch_bin.as_ptr(), epoch_program.as_ptr(), program_size);
    }
}


pub fn miner_testlib_save_program_as_bin(epoch_program: &[u8], program_size: usize)
{
    unsafe  {
        testlib_save_program_as_bin(epoch_program.as_ptr(), program_size);
    }
}


///////////////////////

    
pub fn miner_get_num_devices() -> i32 {

    unsafe {
        cuminer_getNumDevices()
    }
}

pub fn miner_enum_devices() {
    unsafe {
        cuminer_enumDevices();
    }
}


pub use cuinterface::*;


