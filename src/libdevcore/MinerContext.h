#pragma once

#include <cstdint>
#include "Log.h"
#include "AleoCallBack.h"

extern "C" {

/*创建miner 实例 , 返回值为 miner 实例的 指针 ， 这个api 只需要调用一次

最多支持 一个机器 8张 nvidia 的显卡

mask 的默认值是 0xff == 0b11111111 , 如果不调用 该 api ， 那么 mask 就是默认值 0xff

如果 mask = 0b00000001 表示只启用 第 0 张显卡 
如果 mask = 0b00000011 表示只启用 第 0,1 张显卡 
如果 mask = 0b10000011 表示只启用 第 0,1,7 张显卡 

*/
void *aleo_miner_create(simple_log_callback_t cb_log, submit_callback_t cb_submit, uint8_t miner_mask);

//开始挖矿  ， 这个api 只需要调用一次
bool aleo_miner_start(void *ctx);

// 发送新的挖矿 指令 , start_counter 建议设置为0 ，会随机选择一个 counter 开始迭代挖矿
void aleo_miner_set_new_work(void *ctx, uint8_t *epoch, uint8_t *address, uint64_t target, uint64_t start_counter, void *program, size_t program_size);

/*
查询所有显卡的在 某个时间间隔内计算出的 solution 数量

譬如： 
开始挖矿的时间是 t0 

第1次调用本api 的时间是 t1, 那么第1次调用的结果就是 t1-t0 秒内 计算的hash 数

第2次调用本api 的时间是 t2, 那么第2次调用的结果就是 t2-t1 秒内 计算的hash 数

依次类推。

参数: 
ctx : 懂的都懂
sol : 存放hash值的内存数组， 调用者必须提前分配好一个长度至少为 8的 uint32数组
sol_len : 这是一个 in/out 值， 存放返回的显卡的数量

返回值： 

如果有 4个 显卡，那么返回的时候，会在 sol 内依次填充4个显卡的hash 数据

同时，会把 sol_len 的值设置为4 


譬如:
uint32_t sol_len = 8
uint32_t *sol = malloc(sizeof(uint32_t)*sol_len);


aleo_miner_query_hash(ctx, sol, &sol_len);

*/
void aleo_miner_query_hash(void *ctx, uint32_t *sol , uint32_t *sol_len);


}

