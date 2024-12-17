import sys

def get_data(filename, to_index):
    pub_data = []
    cnt = 0
    with open(filename, "r") as fp:
        lines = fp.readlines()
        oline = 0
        for line in lines:
            oline += 1
            line = line.strip()
            if line.startswith("public=[") or line.startswith("private=[") or line.startswith("leaf_hash=") :
                cnt += 1
                if cnt == to_index :
                    print(f"line in orig file is {oline}")
                    break
                
                    


    return pub_data


get_data(sys.argv[1], int(sys.argv[2]))