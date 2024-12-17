import sys, os

def bigint_to_u64_array(big_int):
    mask = (1 <<64) -1 
    code = ""
    for n in range(0, 4):
        v = ( big_int >> (n*64) ) & mask 
        code += "0x{:016x}ULL , ".format(v) 
    
    print(code)




if sys.argv[1] == "bigint_u64":
    print("conver bigint to u64 array")
    bigint_to_u64_array( int(sys.argv[2]) )
