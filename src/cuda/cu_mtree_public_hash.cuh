#pragma once


#include <stdio.h>
#include "cu_synthesis.cuh"
#include "cu_mtree_leaf.cuh"

__device__ uint64_t svm_generate_constant_from_rand(SVM_PARAM_DEF, uint64_t counter);