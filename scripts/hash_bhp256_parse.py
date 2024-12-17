import os,sys
from bitarray import bitarray

from bitarray.util import ba2int


def field_to_uint64_arr(field_int):
    result = [] 
    mask = (1<<64) -1 
    for n in range(0,4):
        v = field_int & mask 
        result.append( '0x%016xULL' % v )
        field_int = field_int >> 64
    result = ', '.join(result)
    code = " { .uint64 =  { " + result + " }  }, "
    return code

class ParseBhp256:
    def __init__(self, logfilename):
        self.filename = logfilename
        self.result = {}
        self.bases = []
        self.domain = bitarray()

    def to_int_value(self, txt_value : str):
        return int(txt_value)

    def parse_with_base_line(self, line):        
        for v in line.split():
            self.bases.append( self.to_int_value(v) )

    def parse_base(self,line):
        pass

    def parse_all(self):
        with open(self.filename,"r") as fp:
            all_lines = fp.readlines()
        
        for line in all_lines:
            line = line.strip()            
            self.parse_with_base_line(line)
            

p = ParseBhp256(sys.argv[1])

p.parse_all()

# for k in p.result:
#     print(f"{k}={field_to_uint32_arr( p.result[k])}")

for base in p.bases:
    print(f"{field_to_uint64_arr(base)}")

# print(p.domain)
# print(len(p.domain))
#print( field_to_uint32_arr( ba2int(p.domain)) )
