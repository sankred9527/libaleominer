from itertools import product

def Rho_Pi():
    KECCAK_ROTATION = [
        [ 0, 36,  3, 41, 18],
        [ 1, 44, 10, 45,  2],
        [62,  6, 43, 15, 61],
        [28, 55, 25, 21, 56],
        [27, 20, 39,  8, 14]
    ]
    
    for y, x in product(range(5), repeat=2):
        # b[y][(x * 2 + y * 3) % 5] = rotl64(a[x][y], KECCAK_ROTATION[x][y]);
        print(f"SHA3ST[{y*5 + (x * 2 + y * 3) % 5}].stdint = rotl64(SHA3ST[{x*5+y}].stdint, {KECCAK_ROTATION[x][y]});")

def Chi():
    #SHA3ST[x][y].nv_int = chi( b[x][y].nv_int,  b[(x + 1) % 5][y].nv_int,  b[(x + 2) % 5][y].nv_int );
    for x, y in product(range(5), repeat=2):        
        print(f"SHA3ST[{x*5+y}].nv_int = chi( SHA3ST[{x*5+y}].nv_int,  SHA3ST[{((x + 1) % 5 *5) + y}].nv_int,  SHA3ST[{((x + 2) % 5 *5) + y}].nv_int);" )

def Chi2():
    #SHA3ST[x][y].nv_int = chi( b[x][y].nv_int,  b[(x + 1) % 5][y].nv_int,  b[(x + 2) % 5][y].nv_int );
    for y in  range(0,5):
        print(f"u = SHA3ST[{y}].nv_int;")
        print(f"v = SHA3ST[{5+y}].nv_int;")

        for x in range(0,3):
            #print(y,x)            
            print(f"SHA3ST[{x*5+y}].nv_int = chi( SHA3ST[{x*5+y}].nv_int,  SHA3ST[{((x + 1) % 5 *5) + y}].nv_int,  SHA3ST[{((x + 2) % 5 *5) + y}].nv_int);" )
        # x == 3
        print(f"SHA3ST[{3*5+y}].nv_int = chi( SHA3ST[{3*5+y}].nv_int,  SHA3ST[{4*5+y}].nv_int,  u);" )
        # x == 4
        print(f"SHA3ST[{4*5+y}].nv_int = chi( SHA3ST[{4*5+y}].nv_int,  u,  v);" )

def update_macro():
    code = "#pragma once\n\n"
    for r in range(0,4):
        for offset in range(0,8):
            code += f"#define UP_R_{r}_OFFSET_{offset}(sha3_state,last_bit,input) KeccakV2::fast_path_hash_update_r0_to_r3_by_u32<{r},{offset}>(sha3_state, last_bit, input) \n"
    for offset in range(0,8):
        code += f"#define UP_R_4_OFFSET_{offset}(sha3_state,last_bit,input) KeccakV2::fast_path_hash_update_r4_by_u32<{offset}>(sha3_state, last_bit, input) \n"

    for r in range(5,8):
        for offset in range(0,8):
            code += f"#define UP_R_{r}_OFFSET_{offset}(sha3_state,last_bit,input) KeccakV2::fast_path_hash_update_r5_to_r7_by_u32<{r},{offset}>(sha3_state, last_bit, input) \n"
    print(code)

def main():
    #Rho_Pi()
    update_macro()
    #Chi2()

def sha3_25_param_def():
    code =""
    for n in range(0,25):
        code += f"uint64_t v{n}"
        if n != 24:
            code += ", "
    print(code)

def sha3_25_param():
    code =""
    for n in range(0,25):
        code += f"gl_r0_st[{n}].stdint"
        if n != 24:
            code += ", "
    print(code)

if __name__ == "__main__":
    #main()
    sha3_25_param()

