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

def sha3_final():
    # output.bytes[i] = static_cast<uint8_t>(state[5*((i>>3) % 5)].stdint >> ((i & 7) << 3));
    for i in range(0,32):
        print(f"output.bytes[{i}] = static_cast<uint8_t>(state[{5*((i>>3) % 5)}].stdint >> {(i & 7) << 3} );");

def main():
    #Rho_Pi()
    #Chi2()
    sha3_final()

if __name__ == "__main__":
    main()
