#!/usr/bin/env python3
import ast
import sys,os


def get_data(filename):
    pub_data = []
    with open(filename, "r") as fp:
    
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("public=[") or line.startswith("private=[") or line.startswith("leaf_hash=") :
                s1 = line.find('[') 
                s2 = line.find(']')
                if ( s1>=0 and s2>=0 ):
                    
                    v = ast.literal_eval( line[s1:(s2+1)] )
                    pub_data.append(v)
                    


    return pub_data


def do_comapre(v1,v2, cmp_size):
    if len(v1) != len(v2): 
        print(f"len wrong: {len(v1)}, {len(v2)}");
        if cmp_size <=0:
            return False

    for n in range(0, len(v1)):
        if (cmp_size > 0 ) and n >= cmp_size :
            break

        if v1[n] == v2[n]:
            pass
        else:
            print(f"index {n} is not eq")
            return False
    return True

hash1 = get_data(sys.argv[1])

_hash2 = get_data(sys.argv[2])

if len(_hash2) >= 3072:
    hash2 = _hash2[0:512] + _hash2[512:3072]*5
    if len(_hash2) > 3072:
        hash2 += _hash2[3072:]
else:
    hash2 = _hash2

if len(sys.argv) == 4 :
    cmp_size = int(sys.argv[3])
else:
    cmp_size = -1

if do_comapre(hash1, hash2, cmp_size):
    print("hash is equal")

