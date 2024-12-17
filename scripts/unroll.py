
# #define MTREE_ITER_8_CODE(_ECODE)\
#     _ECODE(0)\
#     _ECODE(1)\
#     _ECODE(2)\
#     _ECODE(3)\
#     _ECODE(4)\
#     _ECODE(5)\
#     _ECODE(6)\
#     _ECODE(7)

def build_unroll(iter_cnt):
    code = f"#define ALEO_ITER_{iter_cnt}_CODE(_CREATE_CODE)\\\n"
    for n in range(0,iter_cnt-1):
        code += f"_CREATE_CODE({n});"
        if (n+1)%8 == 0 :
            code += "\\\n"

    code += f"_CREATE_CODE({iter_cnt-1});\n\n"
    print(code)

def build_unroll_param_two(iter_cnt):
    code = f"#define ALEO_ITER_{iter_cnt}_CODE_WITH_2_PARAMS(_CREATE_CODE, p1)\\\n"
    for n in range(0,iter_cnt-1):
        code += f"_CREATE_CODE(p1,{n});"
        if (n+1)%8 == 0 :
            code += "\\\n"

    code += f"_CREATE_CODE(p1,{iter_cnt-1});\n\n"
    print(code)    

#build_unroll(32)

build_unroll_param_two(8)