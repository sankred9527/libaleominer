use std::slice;
use std::str::FromStr;
use std::convert::TryInto;

use ledger_puzzle_epoch::EpochProgram;
use snarkvm::console;
use snarkvm::synthesizer::program::{InstructionTrait, StackProgram};
use snarkvm::synthesizer::process;

use snarkvm::console::network::{Network};
use snarkvm::synthesizer::Instruction;
use snarkvm::synthesizer::program::Operand::{Register, Literal };
use snarkvm::synthesizer::program::{ CastType };
use console::program::{PlaintextType, LiteralType, RegisterType };
use console_program::RegisterType::Plaintext;
use console_program::ToBytes;
use snarkvm::circuit::prelude::num_traits::{ToPrimitive};


#[repr(u8)]
#[derive(Debug)]
#[allow(dead_code)]
enum OperandType {
    Reg = 0,
    Constant
}

#[repr(C)]
pub struct VmOperand<const N: usize>  {
    size : u8,
    op_type : u8, // 操作数为 寄存器或者常量
    var_type : u8,  // 操作数类型, i16, u32, u128等。 如果是寄存器，这里固定为 0xFF
    pad : [u8;5],  // data 必须按 8字节对齐
    data: [u8; N],
}

//总共8个字节
#[repr(C)]
pub struct VmInstruction {
    size : u32,
    operator_id : u8, //是 iseq , add.w , hash.ped64
    operands_num : u8, // 有几个操作数
    as_type_id : u8, // 结果存为什么类型
    dest : u8, // 目的寄存器序号
}



#[repr(u8)]
#[derive(Debug)]
#[allow(dead_code)]
enum CircuitTypesID {
    Boolean = 0,
    I8 = 1,
    U8 = 2,
    I16 = 3,
    U16 = 4,
    I32 = 5,
    U32 = 6,
    I64 = 7,
    U64 = 8,
    I128 = 9,
    U128 = 10,
    Field = 11,
    Unknown = 255,
}



#[repr(u8)]
enum VmOperator {
    Neg = 0,
    AbsWrapped,
    Not,
    Xor,
    Nor,
    Nand,
    Or,
    And,
    IsEq,
    IsNotEq,
    Add,
    AddWrapped,
    SubWrapped,
    HashPsd2,
    HashBhp256,
    HashPed64,
    Mul,
    MulWrapped,
    Lt,
    Lte,
    Gt,
    Gte,
    CastLossy,
    ShlWrapped,
    ShrWrapped,
    Ternary,
    Square,
    Pow,
    PowWrapped,
    DivWrapped,
    RemWrapped,
    Modulo,
    Div,
    Inv,
}

fn operand_to_vec<const N: usize>(operand: &VmOperand<N>) -> Vec<u8>
{
    let struct_ptr = operand as *const _ as *const u8;
    let byte_slice: &[u8] = unsafe { core::slice::from_raw_parts(struct_ptr, std::mem::size_of_val(operand)) };
    let mut combined = Vec::with_capacity(byte_slice.len());
    combined.extend_from_slice(byte_slice);
    combined
}

fn instruction_to_vec(inst : &VmInstruction, operand_bin: &Vec<u8>) -> Vec<u8>
{
    let struct_ptr = inst as *const _ as *const u8;
    let byte_slice: &[u8] = unsafe { core::slice::from_raw_parts(struct_ptr, std::mem::size_of_val(inst)) };
    let mut combined = Vec::with_capacity(byte_slice.len() + operand_bin.len());
    combined.extend_from_slice(byte_slice);
    combined.extend(operand_bin);
    combined
}

fn to_8bytes_align_mem<const N : usize>(src: &Vec<u8>) -> [u8;N]
{
    let mut bindata = [0u8;N];
    let pad = N - src.len() ;
    if pad >= 0 {
        bindata[..src.len()].copy_from_slice(src);
    }
    bindata
}

macro_rules! create_operand {
    ($v2:expr, $IntType:ident, $operand_bin:ident) => {{

        let mut operand = VmOperand {
            size: 0,
            op_type: OperandType::Constant as u8,
            var_type: CircuitTypesID::$IntType as u8,
            pad : [0u8;5],
            data: $v2
        };
        operand.size = std::mem::size_of_val(&operand) as u8;
        let slice = operand_to_vec(&operand);
        $operand_bin.extend(slice);
    }};
}

macro_rules! create_instruction {
    ($ret:expr, $operator:ident, $inst:expr, $as_type:expr, $operand_bin:expr) => {{
            let dest = $inst.destinations()[0].locator();
            let mut inst_code = VmInstruction {
                size : std::mem::size_of::<VmInstruction>() as u32,
                operator_id : VmOperator::$operator as u8,
                operands_num : $inst.operands().len() as u8,
                as_type_id : $as_type as u8,
                dest : dest as u8
            };
            inst_code.size += $operand_bin.len() as u32;
            $ret = instruction_to_vec(&inst_code, $operand_bin);
    }};
}

fn convert_register_type_to_my_id<N:Network>(dest_type : &PlaintextType<N>) -> CircuitTypesID
{
    match dest_type {
        PlaintextType::Literal(LiteralType::I8) => { CircuitTypesID::I8 }
        PlaintextType::Literal(LiteralType::U8) => { CircuitTypesID::U8 }

        PlaintextType::Literal(LiteralType::I16) => { CircuitTypesID::I16 }
        PlaintextType::Literal(LiteralType::U16) => { CircuitTypesID::U16 }

        PlaintextType::Literal(LiteralType::I32) => { CircuitTypesID::I32 }
        PlaintextType::Literal(LiteralType::U32) => { CircuitTypesID::U32 }

        PlaintextType::Literal(LiteralType::I64) => { CircuitTypesID::I64 }
        PlaintextType::Literal(LiteralType::U64) => { CircuitTypesID::U64 }

        PlaintextType::Literal(LiteralType::I128) => { CircuitTypesID::I128 }
        PlaintextType::Literal(LiteralType::U128) => { CircuitTypesID::U128 }

        PlaintextType::Literal(LiteralType::Field) => { CircuitTypesID::Field }

        _ => { CircuitTypesID::Unknown }
    }
}


fn convert_one_instruction_to_binary<N:Network>(epoch_program : &EpochProgram<N>, instruction : &Instruction<N>) -> Vec<u8> {

    let function_name = console::program::Identifier::from_str("synthesize").unwrap();
    let function = epoch_program.stack().get_function(&function_name).unwrap();

    let mut operand_types = Vec::with_capacity(instruction.operands().len());
    let reg_types = process::RegisterTypes::from_function(epoch_program.stack(), &function ).unwrap();
    for operand in instruction.operands() {
        // Retrieve and append the register type.
        operand_types.push(reg_types.get_type_from_operand(epoch_program.stack(), operand).unwrap());
    }
    // Compute the destination register types.
    //我们只考虑有1个 dest 的简单情况
    let destination_type = instruction.output_types(epoch_program.stack(), &operand_types).unwrap()[0].clone();

    let mut operand_bin = vec![0u8; 0];
    for  operand in instruction.operands() {
        match &operand {
            Register(v) => {
                let bin_data = to_8bytes_align_mem::<8>(&Vec::from([v.locator() as u8]));
                let mut operand = VmOperand {
                    size: 0,
                    op_type: OperandType::Reg as u8,
                    var_type: CircuitTypesID::Unknown as u8,
                    pad : [0u8;5],
                    data: bin_data
                };
                operand.size = std::mem::size_of_val(&operand)  as u8;
                let slice = operand_to_vec(&operand);
                operand_bin.extend(slice);
                //println!("find operand = {:?}, size={}", operand_bin, operand.size);
            }
            //字面常量
            Literal(v) => {
                match v {
                    console::program::Literal::I8(v2) => {
                        let bin_data = to_8bytes_align_mem::<8>(&v2.to_bytes_le().unwrap());
                        create_operand!(bin_data, I8, operand_bin);
                    }
                    console::program::Literal::U8(v2) => {
                        let bin_data = to_8bytes_align_mem::<8>(&v2.to_bytes_le().unwrap());
                        create_operand!(bin_data, U8, operand_bin);
                    }
                    console::program::Literal::I16(v2) => {
                        let bin_data = to_8bytes_align_mem::<8>(&v2.to_bytes_le().unwrap());
                        create_operand!(bin_data, I16, operand_bin);
                    }
                    console::program::Literal::U16(v2) => {
                        let bin_data = to_8bytes_align_mem::<8>(&v2.to_bytes_le().unwrap());
                        create_operand!(bin_data, U16, operand_bin);
                    }
                    console::program::Literal::I32(v2) => {
                        let bin_data = to_8bytes_align_mem::<8>(&v2.to_bytes_le().unwrap());
                        create_operand!(bin_data, I32, operand_bin);
                    }
                    console::program::Literal::U32(v2) => {
                        let bin_data = to_8bytes_align_mem::<8>(&v2.to_bytes_le().unwrap());
                        create_operand!(bin_data, U32, operand_bin);
                    }
                    console::program::Literal::I64(v2) => {
                        let bin_data = to_8bytes_align_mem::<8>(&v2.to_bytes_le().unwrap());
                        create_operand!(bin_data, I64, operand_bin);
                    }
                    console::program::Literal::U64(v2) => {
                        let bin_data = to_8bytes_align_mem::<8>(&v2.to_bytes_le().unwrap());
                        create_operand!(bin_data, U64, operand_bin);
                    }
                    console::program::Literal::I128(v2) => {
                        let bin_data = to_8bytes_align_mem::<16>(&v2.to_bytes_le().unwrap());
                        create_operand!(bin_data, I128, operand_bin);
                    }
                    console::program::Literal::U128(v2) => {
                        let bin_data = to_8bytes_align_mem::<16>(&v2.to_bytes_le().unwrap());
                        create_operand!(bin_data, U128, operand_bin);
                    }
                    console::program::Literal::Boolean(v2) => {
                        let mut operand = VmOperand {
                            size: 0,
                            op_type: OperandType::Constant as u8,
                            var_type: CircuitTypesID::Boolean as u8,
                            pad : [0u8;5],
                            data: [ **v2 as u8 ; 8]
                        };
                        operand.size = std::mem::size_of_val(&operand) as u8;
                        let slice = operand_to_vec(&operand);
                        operand_bin.extend(slice);
                    }
                    console::program::Literal::Field(v2) => {
                        let bin_data = to_8bytes_align_mem::<32>(&v2.to_bytes_le().unwrap());
                        create_operand!(bin_data, Field, operand_bin);
                    }
                    _ => {
                        panic!("wrong literal constant={:?}", v);
                    }
                }

            }
            _ => {
                panic!("not support operand type: {:?}", operand);
            }
        }
    }

    //返回整个 instruction 对应的 bytes stream
    let mut ret = vec![0u8;0];
    match instruction {
        Instruction::CastLossy(v) => {
            let vtype = match v.cast_type() {
                CastType::Plaintext(ct) => {
                    convert_register_type_to_my_id(ct)
                }
                _ => { panic!("wrong 2")}
            };
            create_instruction!(ret, CastLossy, &instruction, vtype, &operand_bin);
        }
        Instruction::HashPSD2(v) => {
            let instToId = match &destination_type {
                Plaintext(pv) => {
                    convert_register_type_to_my_id(pv)
                }
                _ => { panic!("wrong 1") }
            };
            create_instruction!(ret, HashPsd2, &instruction, instToId, &operand_bin);
        }
        Instruction::HashPED64(v) => {
            let instToId = match &destination_type {
                Plaintext(pv) => {
                    convert_register_type_to_my_id(pv)
                }
                _ => { panic!("wrong 1") }
            };
            create_instruction!(ret, HashPed64, &instruction, instToId, &operand_bin);
        }
        Instruction::HashBHP256(v) => {
            let instToId = match &destination_type {
                Plaintext(pv) => {
                    convert_register_type_to_my_id(pv)
                }
                _ => { panic!("wrong 1") }
            };
            create_instruction!(ret, HashBhp256, &instruction, instToId, &operand_bin);
        }
        Instruction::AbsWrapped(v) => {
            create_instruction!(ret, AbsWrapped, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::IsEq(v) => {
            create_instruction!(ret, IsEq, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::IsNeq(v) => {
            create_instruction!(ret, IsNotEq, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::AddWrapped(v) => {
            create_instruction!(ret, AddWrapped, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Add(v) => {
            create_instruction!(ret, Add, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Not(v) => {
            create_instruction!(ret, Not, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Neg(v) => {
            create_instruction!(ret, Neg, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::SubWrapped(v) => {
            create_instruction!(ret, SubWrapped, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Mul(v) => {
            create_instruction!(ret, Mul, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::MulWrapped(v) => {
            create_instruction!(ret, MulWrapped, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Xor(v) => {
            create_instruction!(ret, Xor, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Nor(v) => {
            create_instruction!(ret, Nor, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Nand(v) => {
            create_instruction!(ret, Nand, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Or(v) => {
            create_instruction!(ret, Or, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::And(v) => {
            create_instruction!(ret, And, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Ternary(v) => {
            create_instruction!(ret, Ternary, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::ShrWrapped(v) => {
            create_instruction!(ret, ShrWrapped, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::ShlWrapped(v) => {
            create_instruction!(ret, ShlWrapped, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::LessThan(v) => {
            create_instruction!(ret, Lt, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::LessThanOrEqual(v) => {
            create_instruction!(ret, Lte, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::GreaterThan(v) => {
            create_instruction!(ret, Gt, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::GreaterThanOrEqual(v) => {
            create_instruction!(ret, Gte, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Square(v) => {
            create_instruction!(ret, Square, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Pow(v) => {
            create_instruction!(ret, Pow, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::PowWrapped(v) => {
            create_instruction!(ret, PowWrapped, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::DivWrapped(v) => {
            create_instruction!(ret, DivWrapped, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::RemWrapped(v) => {
            create_instruction!(ret, RemWrapped, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Modulo(v) => {
            create_instruction!(ret, Modulo, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Div(v) => {
            create_instruction!(ret, Div, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        Instruction::Inv(v) => {
            create_instruction!(ret, Inv, &instruction, CircuitTypesID::Unknown, &operand_bin);
        }
        _ => {
            panic!("not support instruction: {:?}", instruction);
        }
    }
    ret
}

pub fn convert_epoch_program_to_cuda_struct<N:Network>(epoch_program : &EpochProgram<N>) -> Vec<u8> {


    let function_name = console::program::Identifier::from_str("synthesize").unwrap();
    // Retrieve the function from the program.
    let function = epoch_program.stack().get_function(&function_name).unwrap();

    let mut inst_binary = vec![0u8; 0];
    for instruction in function.instructions() {
        inst_binary.extend(convert_one_instruction_to_binary(&epoch_program, &instruction));
    }
    inst_binary
}