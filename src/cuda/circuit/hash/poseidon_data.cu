#include "cuda/cu_bigint_define.cuh"

//预定义的常量

//3x3=9 个
__constant__ bigint_u256_t PSD2_MDF_RAW[9] = {
{ .uint8 = { 0xbe,0xfb,0x7,0xcf,0xf0,0x4,0x37,0x7a,0x50,0x88,0x2d,0x1f,0xfa,0xa8,0xe7,0xd0,0xb4,0x64,0x6d,0xcd,0x9f,0x16,0xe,0x4f,0x4a,0xb3,0x8a,0x98,0x89,0xc5,0x78,0xd } },
{ .uint8 = { 0x86,0xe4,0x6a,0xa7,0x1b,0xc7,0x7f,0x28,0x8,0xfe,0xcd,0x5b,0x72,0x84,0x82,0x26,0xef,0xef,0x7b,0x23,0xfb,0xba,0xaa,0x51,0xcd,0x5c,0xc4,0xa4,0x43,0xec,0x31,0xd } },
{ .uint8 = { 0xde,0x42,0x4a,0xd7,0xcf,0xee,0xed,0x22,0x3c,0x59,0x8d,0x2a,0xf3,0xdf,0x62,0x9b,0x20,0xb4,0x2a,0xb3,0xef,0xca,0xd3,0x45,0xac,0x64,0x1b,0x45,0x90,0xd7,0x6e,0x2 } },
{ .uint8 = { 0xf0,0x6,0x5f,0x4,0x46,0x3a,0xf6,0xbe,0xb9,0x1c,0x69,0x54,0xe,0x44,0xea,0x6a,0x50,0x54,0x4e,0xb9,0x93,0x3c,0x66,0xab,0x78,0x46,0x97,0x2b,0xa6,0xd,0xfd,0x6 } },
{ .uint8 = { 0x24,0x9f,0xc3,0x8d,0xeb,0xa2,0x15,0x7f,0xc8,0xfe,0x9b,0x3,0x63,0xe8,0x0,0x3c,0xf4,0xf9,0x32,0x57,0x68,0x26,0x9b,0xf,0xdd,0x7f,0xe8,0x29,0xbe,0x75,0x2b,0x5 } },
{ .uint8 = { 0x25,0xda,0x2a,0xc6,0x24,0x47,0x86,0x5a,0xa8,0xb1,0x89,0x3d,0xa2,0x5a,0x14,0x75,0x75,0xd3,0x4d,0xe9,0xd2,0xdf,0xb,0xb3,0x78,0x72,0xcf,0x20,0x4,0x1f,0x6,0x7 } },
{ .uint8 = { 0x3d,0x64,0xc7,0x3c,0x76,0x3f,0x97,0xdb,0x66,0xaf,0x87,0xc4,0xc9,0x69,0x56,0x43,0x84,0xf0,0x4f,0xf0,0x1a,0x3b,0xa9,0x5e,0xa,0x9c,0x47,0xe6,0x28,0x5d,0xd,0x5 } },
{ .uint8 = { 0x1d,0xca,0xed,0x18,0xff,0x3a,0x5b,0x42,0x47,0xef,0x2a,0x75,0xd,0x79,0x59,0x68,0x6c,0x2e,0x1c,0xec,0xd7,0xbb,0x28,0x87,0xb6,0x3d,0xc1,0xd6,0x94,0xc1,0x2f,0x0 } },
{ .uint8 = { 0x2e,0x7e,0xd8,0x41,0x4d,0x2b,0xe0,0x5,0x3b,0xb6,0xcd,0x46,0xb8,0x70,0x43,0x79,0x55,0xf7,0xc5,0xec,0x23,0xd7,0xa1,0x76,0x69,0x8c,0x2e,0xfb,0x72,0xd0,0x4e,0x12 } },
};

// 39x3=117
__constant__  bigint_u256_t PSD2_ARK_RAW[117] = {

{ .uint8 = { 0xc3,0xc7,0x16,0x33,0xa2,0x2,0xdb,0xfe,0x10,0x49,0x3b,0x2f,0xe9,0x6b,0x9f,0xd7,0x26,0x8a,0x79,0xfe,0xa4,0x92,0xf4,0x9,0x72,0x6c,0xe2,0xbe,0x80,0xd4,0x7,0x3 } },
{ .uint8 = { 0x6e,0x83,0xba,0xb6,0xfb,0xfb,0xe,0x23,0xb3,0x9c,0xd0,0x54,0x96,0xbe,0x6d,0x1e,0xaa,0x3d,0x8d,0x83,0x6b,0xa0,0x2a,0x14,0xcb,0x2c,0x53,0x4d,0x2d,0xd8,0x54,0xa } },
{ .uint8 = { 0xb8,0x0,0xcb,0xa7,0xe7,0xee,0xe5,0x17,0xf,0x75,0x1c,0xca,0x65,0x78,0xcc,0x5e,0x3f,0xee,0x1c,0x82,0xd4,0xf5,0xd0,0xc4,0xc3,0xa9,0xe6,0x13,0x9d,0xe7,0xaf,0x7 } },
{ .uint8 = { 0x11,0x4c,0x5d,0xa4,0xa0,0xb1,0x36,0x9f,0x2d,0x10,0x62,0x80,0x44,0x95,0x8c,0x2d,0x4c,0x4d,0xd6,0x81,0x4c,0xba,0xd1,0x5c,0x2,0x97,0x3e,0x71,0xad,0xab,0x64,0x2 } },
{ .uint8 = { 0x8e,0xe9,0x9a,0x8b,0xed,0x83,0x33,0x47,0x0,0xef,0x48,0xe,0xb4,0x9b,0x22,0xd3,0xb4,0x5c,0xd,0x7e,0xfa,0xb3,0x21,0x7f,0xb9,0x39,0xd4,0x1a,0x8f,0x47,0xf1,0xa } },
{ .uint8 = { 0x54,0x56,0xce,0x8d,0xe7,0x62,0x50,0x55,0x1c,0x34,0x2a,0xa0,0xfb,0x22,0x55,0x7f,0x59,0x71,0x5d,0xd3,0xd8,0xe6,0x3f,0x4a,0xb7,0xe5,0xb1,0xe,0xce,0x2a,0xf1,0x2 } },
{ .uint8 = { 0xb,0xdf,0xcd,0xd,0x98,0xf9,0x6,0xf7,0xd8,0x8a,0xfe,0x58,0xc3,0x2a,0xbd,0x86,0x1a,0x26,0x63,0xba,0xa,0x51,0x68,0x2f,0x67,0x95,0x28,0x67,0x18,0x91,0xdb,0x5 } },
{ .uint8 = { 0x96,0x60,0x9f,0xbc,0xbf,0x77,0xef,0x7d,0xf2,0x3e,0x6a,0xf7,0xb4,0xef,0x6a,0x15,0xd8,0xc0,0x67,0xb1,0x2b,0x2e,0xd6,0xd1,0x30,0xb9,0xb6,0x17,0xa6,0x97,0xfc,0xa } },
{ .uint8 = { 0x5b,0x7c,0x3e,0x21,0x19,0x6e,0x11,0x15,0xa9,0x1,0xcd,0xfa,0xde,0xd0,0xe4,0xe6,0x8d,0xad,0x8d,0x65,0xbe,0xfd,0x56,0xc2,0x17,0x38,0xd3,0x7f,0xbc,0x62,0x74,0x0 } },
{ .uint8 = { 0x87,0x8a,0x6c,0x9e,0xd1,0x84,0xaf,0x29,0xd3,0x9e,0x28,0xcd,0x29,0x78,0x1a,0x61,0xf4,0xc,0x24,0x95,0x3f,0xae,0xc4,0x0,0xa5,0x9e,0x3b,0x19,0x51,0x62,0x96,0xd } },
{ .uint8 = { 0x61,0x30,0x3d,0xaa,0x3c,0xf2,0xf,0xf5,0xda,0x2d,0xc6,0x81,0xeb,0x80,0x5b,0xeb,0xfd,0x34,0x5e,0xf8,0x2,0xb4,0xf4,0x94,0x72,0xbe,0xd3,0x7d,0x4b,0x3d,0x97,0xc } },
{ .uint8 = { 0x7,0x8f,0x4f,0x1,0x9f,0x7e,0xf0,0xa4,0x65,0xe1,0x3c,0x85,0x5,0x80,0x27,0xaa,0x16,0xe5,0x13,0x7e,0x14,0x20,0xe6,0xec,0x3b,0x76,0x59,0x5c,0xa1,0xb3,0xb1,0xc } },
{ .uint8 = { 0x8e,0x58,0x68,0xf8,0xcc,0xad,0x95,0xde,0xeb,0xe7,0xbd,0xc3,0xd4,0x6c,0x89,0xbb,0x32,0x56,0x7b,0x21,0xf,0xff,0x8c,0x65,0x7f,0x35,0x2f,0xf7,0x81,0xaf,0x4d,0xe } },
{ .uint8 = { 0x16,0x0,0x81,0x89,0xe0,0xee,0xcc,0x69,0x63,0x1f,0xa1,0xe8,0x24,0x38,0x7c,0x7c,0x7,0x18,0x55,0xd4,0xc3,0xc3,0xfa,0xf8,0xcb,0xd6,0x97,0x4d,0x55,0x7,0x2f,0x8 } },
{ .uint8 = { 0xf,0x5d,0x9d,0x8c,0xce,0x9b,0x2f,0xd7,0xb8,0xf3,0x57,0x68,0xd1,0x2a,0xa7,0x45,0x33,0x9d,0x6c,0xdb,0x74,0x40,0xb7,0xc0,0xb,0xfe,0x22,0xad,0xdf,0x14,0x60,0x8 } },
{ .uint8 = { 0xfe,0x8e,0xa3,0xed,0xb2,0xd,0x87,0x99,0x44,0x6d,0x43,0x2b,0xda,0x4b,0xa7,0x85,0xb2,0x64,0x2c,0x0,0x96,0xe4,0xa4,0xec,0xec,0x1e,0xc1,0xca,0xb6,0xf4,0x72,0x10 } },
{ .uint8 = { 0x63,0x43,0xaa,0xea,0x3b,0xb5,0x2d,0xfc,0xc5,0xba,0x1e,0x7a,0xf6,0x7e,0xc1,0x45,0x9c,0x3c,0x42,0xb1,0xd2,0x16,0x3e,0xfd,0x14,0x79,0xf,0xa1,0xb3,0x73,0xc0,0x1 } },
{ .uint8 = { 0xab,0x79,0x13,0xc,0x6a,0x9a,0x1d,0xd2,0x4e,0x89,0x66,0xc6,0x44,0x11,0xa5,0x87,0xb6,0xba,0xc8,0x0,0x48,0x62,0xe0,0x0,0x93,0x62,0x6b,0x12,0x53,0xc3,0xc3,0x7 } },
{ .uint8 = { 0xb4,0x80,0x27,0x37,0x35,0xff,0xb2,0xad,0x6f,0xd1,0x81,0xb2,0x44,0x2,0xd8,0x94,0x8e,0x52,0x71,0x1b,0xf8,0x45,0x7a,0x73,0x44,0x14,0xbd,0x40,0xbe,0x10,0x5b,0x7 } },
{ .uint8 = { 0xf3,0x6f,0xa8,0xdf,0x57,0x28,0xa8,0x81,0x49,0x45,0xeb,0x54,0x80,0xd3,0x6b,0x28,0x76,0x5d,0xf8,0xdd,0xc,0x5c,0x39,0x88,0xf7,0x16,0x48,0xee,0x4e,0x7e,0xd3,0xf } },
{ .uint8 = { 0xaa,0x7,0xe5,0xff,0x4,0xe6,0xd7,0xf,0xdf,0xf1,0xf,0x8f,0x50,0x24,0xa4,0x84,0x6e,0x34,0x7c,0x16,0xc7,0x20,0xca,0x38,0x79,0x3a,0xb0,0xa4,0xe8,0xf2,0x89,0xe } },
{ .uint8 = { 0xb7,0xe0,0xe8,0x7f,0x62,0x77,0xef,0x1a,0xf7,0x62,0x63,0xa3,0x29,0xa0,0x50,0x33,0x15,0xf2,0xce,0x2b,0xc,0xa3,0x66,0x2d,0xc1,0x55,0x8e,0x72,0xf5,0xb6,0xde,0x3 } },
{ .uint8 = { 0xea,0x6f,0xd6,0x58,0x15,0x5,0xda,0x87,0xd9,0x61,0x1b,0xb4,0xcc,0xc6,0xb2,0x8d,0xbb,0x63,0x10,0x32,0xee,0xcc,0x2c,0x4a,0x2d,0x74,0x44,0x75,0x1f,0x2b,0xb0,0xe } },
{ .uint8 = { 0x94,0xf0,0xa5,0xe3,0xfd,0x2f,0xb1,0x9f,0x3b,0xb5,0x1f,0x1a,0xb,0x29,0x50,0xc4,0x33,0xe,0xae,0xc3,0xc,0x6a,0x54,0x6,0x6a,0x20,0x6f,0xc,0xb6,0x62,0x14,0x9 } },
{ .uint8 = { 0xc8,0xe9,0x74,0x31,0x4a,0x8a,0xbf,0x91,0xa4,0x3,0x95,0x54,0xd8,0xc3,0xab,0x88,0x44,0x5a,0xf9,0x2d,0x68,0x21,0xba,0x3d,0xcc,0x5,0x9f,0x1b,0xd1,0xf8,0x7c,0xb } },
{ .uint8 = { 0x2,0x6b,0xb3,0xc0,0x98,0x14,0x2a,0xc1,0xe4,0x9c,0xca,0x31,0xf2,0x71,0xf2,0x3d,0x2d,0x8,0xfb,0xaf,0x85,0x9a,0xf9,0xa5,0xce,0xe7,0x67,0x74,0x75,0xfc,0xf,0xe } },
{ .uint8 = { 0xd1,0x12,0xe7,0x17,0x9d,0x5f,0x9e,0x55,0x59,0x67,0x13,0xb6,0x68,0x52,0xa8,0xc3,0x95,0xbf,0x76,0xbf,0x1,0xf7,0x5d,0x1a,0x70,0x5f,0xd2,0x3a,0x49,0xda,0xa0,0x12 } },
{ .uint8 = { 0x34,0x69,0xfe,0xbb,0x47,0x79,0xc9,0x98,0xcc,0x75,0x88,0xbc,0xc0,0x8,0xbf,0xca,0x32,0xb8,0xec,0x89,0xd3,0x8e,0x6f,0xdb,0x17,0x3f,0x51,0x99,0xa5,0x77,0xbd,0x10 } },
{ .uint8 = { 0xd3,0x79,0x6e,0x74,0xdc,0xd0,0xb0,0xd1,0x3e,0xd8,0xe9,0x16,0x4a,0x45,0xc8,0xf7,0x3f,0xc6,0xd,0x9e,0x1a,0x2,0xef,0x4d,0x27,0x43,0xb8,0x9a,0x18,0xee,0x88,0x2 } },
{ .uint8 = { 0x6b,0xda,0x74,0xa2,0x8c,0xb1,0x36,0x56,0x96,0x8c,0xc0,0xdc,0x3c,0x82,0x2a,0x1f,0x33,0xe9,0xaf,0x9e,0x22,0x8c,0x4e,0x1,0x56,0x7a,0xdb,0x74,0x80,0x7,0x9f,0xf } },
{ .uint8 = { 0x15,0xa7,0x4d,0x98,0xcb,0xf0,0xa0,0x96,0x38,0xc3,0x43,0x7d,0xdd,0x6e,0x62,0x7e,0x8a,0xab,0xa7,0x9c,0x70,0xa6,0xf5,0xa1,0x27,0x74,0xa9,0x73,0x70,0x2d,0xa0,0x6 } },
{ .uint8 = { 0x45,0x77,0x2,0x44,0x19,0xc1,0xe1,0xe,0x5a,0x52,0xc5,0x6b,0xd7,0x72,0x9d,0x61,0x4d,0x56,0x17,0x20,0xb3,0xde,0xb5,0x71,0x8f,0xcf,0xd4,0x81,0x2,0x5b,0xf4,0x10 } },
{ .uint8 = { 0x3d,0xe,0xc4,0xc7,0x3f,0x58,0xac,0xbd,0x39,0xdd,0x4,0xc0,0xfd,0x60,0xf3,0xa7,0x17,0x9a,0xbb,0xd6,0xad,0x61,0xca,0xf9,0xb3,0xbd,0x3a,0x38,0xcc,0x1c,0x12,0x2 } },
{ .uint8 = { 0xe,0xdc,0x9d,0xa9,0xa7,0x63,0xc6,0x2d,0x4d,0x44,0xdc,0x5c,0x85,0x23,0x58,0x5a,0x2b,0x93,0xd5,0xc1,0x6d,0x97,0x43,0x58,0x2c,0x92,0xac,0x92,0x7,0xef,0xb4,0xd } },
{ .uint8 = { 0x9f,0xaa,0xf4,0xd7,0xce,0x19,0xad,0xe9,0xb5,0x1e,0xa9,0x73,0x8e,0xfa,0x49,0xd8,0xa5,0xc,0x21,0x13,0x8f,0xd,0x4a,0x6,0x2d,0x15,0x1a,0x87,0xc2,0xa3,0xa7,0x3 } },
{ .uint8 = { 0x1e,0x24,0xe9,0x6,0x66,0x76,0x23,0xed,0x1,0xc4,0xf7,0xf2,0xa5,0xbf,0xaf,0x79,0x35,0x3,0xe2,0x8c,0xd2,0x2e,0xd6,0x2f,0xfe,0x77,0x9,0x78,0xac,0x84,0x8d,0x11 } },
{ .uint8 = { 0xce,0xbb,0xbd,0x93,0xf6,0x99,0x7c,0x26,0xfc,0x5f,0xc9,0x6c,0x40,0x22,0x6c,0xd5,0xb7,0x66,0xc6,0x68,0x22,0x75,0x80,0xca,0xad,0x6c,0xf6,0x10,0x66,0x7a,0x9b,0xb } },
{ .uint8 = { 0x4e,0x80,0xec,0xfe,0x5e,0xce,0x9b,0xb2,0xb5,0x58,0x7a,0x9c,0x62,0xf7,0xe2,0x89,0xbe,0xe9,0x8b,0xbc,0xc9,0x6f,0xf8,0x3c,0xe9,0xce,0x76,0x75,0x27,0x84,0x85,0x0 } },
{ .uint8 = { 0xef,0x27,0xf,0x1b,0x60,0x87,0x9,0x96,0x1,0xf5,0xfc,0x7a,0xf0,0x41,0x7,0x65,0x64,0x26,0x72,0xa5,0x1b,0x6f,0xb0,0x70,0x3f,0x14,0xf1,0x2d,0x63,0x8a,0x9c,0xe } },
{ .uint8 = { 0xeb,0x49,0xe0,0xd0,0xaa,0x5b,0xd4,0x77,0xf7,0xe3,0xf5,0x1d,0x98,0x17,0x40,0x6,0x1e,0xa6,0xc7,0x79,0x98,0x8c,0x1a,0x27,0x3e,0x48,0xba,0x16,0xef,0xbc,0x84,0x1 } },
{ .uint8 = { 0x62,0xb4,0xa6,0xb3,0xcd,0x1b,0xd4,0x7e,0x81,0x56,0xb4,0xc1,0x19,0xa8,0xa0,0xe4,0xc5,0xfa,0x4b,0xf9,0x82,0x7b,0x17,0x3b,0x1f,0xd4,0x4e,0x49,0xa1,0x8e,0x1d,0xf } },
{ .uint8 = { 0x87,0x4c,0x3a,0xd9,0xd2,0x89,0xf3,0xb8,0x78,0xd1,0x8a,0x40,0x6d,0xf,0x53,0x1a,0xf1,0xcb,0x85,0x8d,0x95,0x46,0xcc,0x63,0x88,0xae,0x3c,0x4c,0xac,0xb6,0xc0,0x9 } },
{ .uint8 = { 0x15,0xe6,0x4,0x9c,0x5f,0x3e,0xf2,0xc3,0xe0,0xed,0x9c,0xbc,0x67,0x87,0x91,0x48,0x6b,0xcb,0xab,0xd,0x5a,0x6e,0x8e,0x57,0x4c,0xff,0xc9,0x2a,0xb3,0x29,0x2b,0x11 } },
{ .uint8 = { 0x62,0xff,0xae,0x4f,0xf8,0xc3,0xbf,0xd3,0xf9,0x4c,0x5e,0xea,0x36,0x15,0xb7,0xef,0xb4,0x5,0x56,0xf,0x64,0x39,0x92,0xa8,0x32,0xcf,0x98,0xf1,0x6b,0x37,0xbc,0x0 } },
{ .uint8 = { 0x46,0x13,0xb4,0xa9,0x7d,0xcf,0x1d,0x1d,0x74,0x63,0x10,0x37,0x92,0x71,0x77,0x31,0x55,0x1,0xc8,0x1c,0x34,0x74,0xe3,0x1a,0x7,0x2,0xeb,0x96,0x9,0x11,0xa8,0x12 } },
{ .uint8 = { 0x69,0x51,0x5a,0xc8,0xf6,0x60,0xa6,0xb7,0x44,0x8a,0xc1,0xa9,0xd8,0x78,0xd1,0xa,0xcc,0x85,0xc1,0xcd,0x59,0xbf,0x3c,0x30,0xca,0x65,0xa4,0x2,0x7c,0x6,0xaa,0x9 } },
{ .uint8 = { 0x2d,0x7e,0xdb,0xc3,0x4f,0x5b,0xb0,0x89,0x99,0x7c,0xd1,0xee,0xb9,0x63,0x3,0xcf,0x5a,0x46,0xd1,0xd1,0xa2,0x1b,0x5,0x39,0x95,0x2f,0xf9,0x60,0xc0,0xf5,0xe2,0x9 } },
{ .uint8 = { 0x31,0x6,0x64,0x7c,0xbf,0xc9,0xe9,0x37,0x4f,0x7f,0x41,0xb4,0x7b,0xc0,0x44,0xb0,0xb,0xa3,0xfa,0x44,0x66,0x1,0x29,0xc6,0x86,0xf4,0x8f,0x7b,0x74,0xa5,0x8a,0x11 } },
{ .uint8 = { 0xf2,0xb0,0xd6,0x46,0x2e,0x3c,0x6,0x92,0xdd,0x5f,0x23,0xd1,0x35,0xec,0x32,0x2a,0x8f,0x86,0xff,0xa5,0xdc,0x89,0x9a,0x27,0xbd,0x3b,0x3d,0x67,0xe5,0xfe,0x49,0x7 } },
{ .uint8 = { 0x12,0xa8,0x2b,0x0,0xf4,0xa5,0x9b,0x2b,0x34,0x67,0xa4,0x20,0x8a,0xec,0xbd,0x0,0x3c,0xf4,0xec,0x9a,0x9c,0xa3,0x1c,0xe5,0x28,0xd4,0xc2,0xb3,0xd9,0xe3,0x4e,0x11 } },
{ .uint8 = { 0x3e,0xc2,0x60,0xd7,0x8,0x4e,0x27,0xae,0xd3,0xe,0xec,0x59,0x7c,0x57,0xa,0x2d,0xa6,0xe,0xcc,0xe,0x6b,0x86,0x34,0x16,0x59,0x19,0x59,0x5d,0xf5,0x3b,0xf5,0xd } },
{ .uint8 = { 0xc5,0xf,0x6a,0x9e,0xca,0x85,0xea,0x69,0x42,0xb,0xd5,0x7e,0x20,0xf4,0x96,0x34,0x14,0xa4,0x54,0xb8,0x57,0x99,0x93,0x15,0x34,0x0,0x19,0x33,0xd1,0xd0,0xf6,0x10 } },
{ .uint8 = { 0x67,0xa,0x30,0x4c,0xe4,0xb8,0x5d,0x83,0xce,0x8f,0x80,0x42,0x69,0x19,0x8c,0xfa,0xd7,0xc6,0x36,0xa6,0xa0,0x71,0xba,0x3f,0xe9,0x19,0xf0,0x0,0xf1,0xc6,0xef,0x7 } },
{ .uint8 = { 0xd8,0x8d,0xaf,0x5c,0x29,0x1a,0x8,0xf2,0x9d,0x1a,0x78,0xc0,0x69,0x41,0xef,0x5d,0x38,0x8f,0xbf,0x1f,0x2d,0xa6,0x14,0xe4,0xf7,0x8b,0xa0,0x27,0x3e,0x68,0xbb,0x7 } },
{ .uint8 = { 0xd3,0xa5,0x5e,0xf1,0xca,0x9c,0xcc,0xcc,0x8,0x3d,0xbb,0x14,0x83,0xaf,0xa,0xa5,0x93,0x6f,0xdd,0xba,0xa,0x9b,0xaf,0x70,0x37,0x5f,0x41,0x76,0xde,0xeb,0xf4,0x7 } },
{ .uint8 = { 0xda,0xd3,0xf2,0x4d,0xc2,0xb8,0x42,0x19,0x52,0x9d,0x78,0xe3,0x45,0x51,0x21,0xf6,0x78,0x1f,0x77,0x3b,0x1e,0x60,0xa4,0x87,0x9c,0xb9,0x76,0xf9,0xb7,0xeb,0x66,0x12 } },
{ .uint8 = { 0xd0,0xa3,0xdb,0x9b,0x8e,0xf5,0xa8,0xc0,0xbe,0x16,0x67,0x52,0xa1,0x27,0x79,0x1f,0x6d,0xe0,0xd3,0xcb,0x68,0xbe,0xc4,0xca,0xe3,0x59,0x25,0x85,0x41,0x3b,0x95,0x5 } },
{ .uint8 = { 0xbb,0xe9,0x7e,0xf,0xca,0xb4,0x9c,0x8,0x28,0x61,0xb8,0x4f,0x42,0x70,0x6c,0xf6,0x64,0xab,0xec,0x5f,0xbf,0xb6,0xc3,0xe3,0x62,0x8b,0x54,0x1d,0xa5,0x41,0xb1,0x3 } },
{ .uint8 = { 0x82,0x2c,0x5,0x2a,0x65,0x9f,0x90,0x7b,0x5a,0xa,0xcf,0xd3,0x59,0x77,0x15,0x64,0x13,0xf6,0xfa,0x73,0x27,0x37,0xac,0xe7,0xa4,0xac,0xc4,0x6f,0xaf,0x99,0xc8,0x5 } },
{ .uint8 = { 0xe8,0x48,0x58,0x80,0xf8,0x6e,0x34,0xaf,0xe0,0x8b,0x10,0x37,0x22,0xa8,0x19,0xf6,0x8c,0xe5,0x1c,0x2f,0x4,0xde,0xb1,0x67,0x3,0x68,0x32,0x6f,0x17,0x8a,0x80,0x8 } },
{ .uint8 = { 0xab,0x9c,0xf8,0x20,0xc5,0x7c,0x96,0x6f,0x6c,0x30,0xb6,0x37,0xe5,0x49,0x2e,0xf3,0x20,0x37,0xd4,0xe,0xfa,0xa4,0x4a,0x2b,0xc8,0x68,0x2a,0xc1,0x91,0x13,0xef,0x11 } },
{ .uint8 = { 0x7,0x50,0x7e,0xd3,0xb1,0xd2,0x43,0x27,0x41,0x14,0xfd,0x7f,0x6e,0x7f,0x56,0x3f,0xd7,0x64,0x23,0x37,0x35,0x56,0xe6,0xd5,0xc3,0x79,0xeb,0x24,0x23,0xf9,0x8f,0x12 } },
{ .uint8 = { 0x33,0xb2,0x7,0x98,0x67,0x21,0x5d,0x7e,0xea,0x78,0x73,0xeb,0xc0,0x70,0xa1,0x3e,0x2a,0xa0,0xad,0x7d,0xdf,0xef,0xa,0x51,0x8,0x12,0xc9,0x9f,0xef,0x6d,0x64,0x10 } },
{ .uint8 = { 0xbb,0xb8,0xd2,0x6f,0xa7,0x7a,0xff,0x22,0x7c,0xdd,0x61,0xb8,0x92,0xab,0x8a,0x33,0x8b,0x9a,0xf,0x49,0x9f,0xc9,0xcb,0x8d,0xb1,0x90,0x73,0x44,0x6d,0x72,0x5a,0x8 } },
{ .uint8 = { 0x5c,0xe7,0x8d,0x9d,0xc5,0x73,0xbb,0x39,0xdf,0x7f,0x15,0xf,0x72,0x23,0x42,0x1b,0x5e,0x7d,0xce,0x98,0x74,0x52,0xba,0x7a,0xcb,0x90,0x73,0x3,0xcc,0xf2,0xab,0x5 } },
{ .uint8 = { 0x2f,0x4c,0xd,0x9e,0x63,0xb0,0x1f,0x63,0xaf,0x2f,0x5b,0xcd,0x77,0x10,0x38,0xbe,0x9f,0xdf,0x5f,0xaa,0x2b,0x7f,0x77,0x5b,0x49,0xce,0x91,0xfc,0x4,0x1d,0x17,0x10 } },
{ .uint8 = { 0xe3,0x78,0x7e,0x73,0x84,0xe5,0x7,0x79,0xc2,0x6d,0x95,0x99,0x94,0x4b,0xe2,0xd5,0x42,0x22,0xd5,0x1,0x25,0x6a,0xdd,0x75,0xe1,0xad,0x9,0xc0,0x43,0xb8,0x71,0xe } },
{ .uint8 = { 0x64,0x8e,0x89,0x90,0x23,0xdd,0xba,0xab,0x5d,0xbc,0xb1,0xc1,0xea,0x96,0xe3,0x85,0xc7,0xe2,0x6d,0xc8,0xc,0xc7,0x6b,0xae,0xff,0xb5,0xe9,0x9c,0xea,0x39,0x87,0xf } },
{ .uint8 = { 0x9a,0xb7,0xbc,0xac,0x2f,0x45,0xff,0x17,0x42,0x32,0x7,0xc9,0x6a,0xe3,0x80,0xaa,0x5f,0x26,0xb0,0x46,0x3e,0xe4,0x56,0x5a,0x68,0x6b,0x66,0x79,0x28,0xee,0x45,0xe } },
{ .uint8 = { 0x4a,0xaa,0xd5,0x7c,0x5c,0xa0,0x89,0x1f,0xda,0x6f,0x64,0xba,0xb5,0x23,0x84,0xa0,0x99,0x14,0xf2,0xcf,0x3c,0xfe,0xe7,0x21,0xf8,0xdc,0xcb,0xd,0x46,0x74,0x26,0x6 } },
{ .uint8 = { 0x81,0xc8,0x26,0x9f,0xda,0x2,0x4a,0xdd,0x7a,0x38,0x91,0xea,0x9c,0xab,0x4a,0x97,0x8c,0xdd,0x3f,0x17,0x9a,0x6,0x9e,0x1a,0xf,0x68,0xf2,0xcc,0xab,0xd,0x2d,0x11 } },
{ .uint8 = { 0x78,0xec,0xcd,0xa9,0x50,0xcc,0xda,0x96,0xcd,0x5f,0x5e,0x20,0xa4,0x1f,0xc1,0xa7,0x72,0xaf,0xd6,0x20,0xb,0x41,0x2b,0x51,0x44,0xf4,0xab,0x8c,0x42,0x3d,0x4a,0xe } },
{ .uint8 = { 0xc5,0x33,0x83,0x5d,0x37,0x37,0x7c,0x2f,0x48,0xc4,0x4b,0xe5,0x3c,0x3e,0x82,0x48,0xba,0xe8,0xea,0xc8,0x1f,0x41,0xac,0xd7,0xf7,0x7d,0xbd,0x78,0x5,0x30,0x70,0x9 } },
{ .uint8 = { 0x85,0xf8,0xf,0xfa,0x5b,0x10,0x8b,0x29,0x9,0x6b,0xc7,0xa1,0x97,0xdc,0x81,0x23,0x83,0xde,0xbc,0x64,0x21,0xd1,0xa1,0xb1,0xd5,0x27,0xcb,0x16,0x36,0xb2,0x6b,0xc } },
{ .uint8 = { 0x7f,0xc4,0xe,0x52,0xf9,0x59,0x7d,0x34,0x18,0x2,0xbf,0xb7,0x5e,0x4,0x89,0x24,0xcf,0x2c,0x63,0x3a,0xf5,0xb5,0x9,0x62,0xe0,0xc4,0x26,0xf8,0x5f,0xab,0x7e,0x2 } },
{ .uint8 = { 0x27,0x4c,0xdf,0xe3,0xb7,0x5f,0xea,0x8f,0xdb,0xa7,0x30,0xb,0x45,0xfa,0xa9,0xf7,0x14,0x1e,0x5d,0x4d,0x5e,0xaa,0x8d,0x31,0x82,0x8a,0x54,0xb2,0x12,0xae,0x41,0xf } },
{ .uint8 = { 0xde,0xf3,0x6b,0xc3,0x3c,0x4d,0x3a,0x8b,0x96,0xb2,0x91,0x6a,0xc8,0x2f,0xf,0xe5,0xf1,0xd7,0xd8,0xe5,0xa9,0x30,0x9,0x1c,0x6f,0x66,0x31,0x59,0xb,0x80,0x56,0x8 } },
{ .uint8 = { 0xe0,0xf4,0xec,0xd5,0x35,0xcf,0x72,0x24,0xdf,0xf1,0x2f,0xa5,0x2f,0xd3,0x2,0x9d,0x18,0xae,0x9b,0xe4,0xed,0x2e,0x39,0xb1,0x21,0xc,0x45,0xea,0x2,0x75,0x11,0x9 } },
{ .uint8 = { 0x9c,0xbf,0x3e,0xae,0xa2,0x95,0xc0,0x6c,0x0,0x47,0xb8,0x20,0xf4,0xc6,0xd1,0xea,0x21,0xfe,0xb9,0xb0,0x9,0xab,0x81,0x21,0x61,0x1b,0x30,0x3a,0x7,0x64,0x10,0x10 } },
{ .uint8 = { 0x4e,0x49,0x7a,0xd0,0xe6,0xb4,0x2b,0xab,0x8e,0x8c,0x22,0xe1,0x84,0x91,0x8e,0x35,0x91,0xfc,0x23,0xee,0xb0,0x7c,0x92,0x7c,0x98,0xce,0x2f,0xe,0x47,0x81,0x89,0xa } },
{ .uint8 = { 0x1b,0xfd,0xd7,0xcf,0x11,0x4,0x14,0xcc,0x59,0xba,0x85,0x18,0x17,0xaf,0x83,0x83,0x5a,0x4f,0x4b,0x7b,0xc8,0xb8,0xa5,0x80,0x8e,0xcd,0x2,0xd7,0xfd,0x81,0x38,0xe } },
{ .uint8 = { 0x73,0x32,0x6a,0x80,0x5c,0x7b,0xc8,0x7,0x29,0x96,0xed,0x40,0xbe,0x90,0x96,0xa,0xee,0x66,0x39,0xd7,0x70,0x46,0x22,0x30,0xa3,0x8c,0x99,0xe0,0xb4,0x67,0x10,0xe } },
{ .uint8 = { 0x30,0x67,0xdd,0x4f,0x81,0xf4,0xef,0xf,0x8d,0x33,0x72,0xd6,0xde,0xef,0x9b,0xb8,0x8f,0xba,0xfd,0xed,0x1d,0x4b,0xe,0x97,0x62,0xc0,0x25,0x85,0x38,0x52,0xd3,0x10 } },
{ .uint8 = { 0xfc,0xdb,0x68,0x11,0xd6,0x88,0xf0,0xd6,0x39,0xfe,0xd1,0xa3,0x68,0xcc,0xba,0xf4,0xa0,0xf4,0x9b,0x92,0xd2,0x47,0x49,0x36,0x8d,0xe9,0x82,0xe0,0x99,0x8d,0xcd,0x0 } },
{ .uint8 = { 0xa3,0x6f,0xb3,0xe5,0x7f,0xc1,0x6c,0xb7,0x56,0xb5,0x5b,0xd5,0x5,0x49,0xfa,0xd0,0x94,0x99,0xef,0x6d,0xeb,0xfb,0x7a,0x72,0x7f,0xa,0x70,0x8,0xba,0x28,0xe0,0x7 } },
{ .uint8 = { 0x94,0xd1,0x2e,0xfa,0x59,0xfb,0x24,0x5c,0x35,0xd,0x79,0x78,0xa,0x54,0xf2,0x99,0xaf,0xc8,0x9a,0x8e,0x1e,0x83,0xe9,0x20,0x2a,0xf4,0x45,0xa0,0x4a,0xc4,0x34,0x11 } },
{ .uint8 = { 0x87,0xf7,0x11,0xe,0x18,0x93,0xbc,0xbe,0x55,0xb6,0x5d,0x5a,0x29,0x6,0xd0,0x40,0x49,0x67,0x43,0xd3,0x87,0xde,0x7f,0xdf,0xdd,0xd0,0x8f,0xa0,0xeb,0xef,0xd1,0x2 } },
{ .uint8 = { 0x94,0x76,0x3,0x23,0xaa,0x68,0x39,0x4d,0xe9,0x3d,0x6c,0x3a,0x39,0x7d,0x9c,0xda,0x7c,0xab,0x9e,0x14,0x59,0xf,0x7b,0x1a,0x3,0xb8,0x4d,0x50,0x86,0xd8,0x48,0xb } },
{ .uint8 = { 0xa6,0x53,0x44,0x92,0x6c,0x5,0x78,0x6e,0xa2,0x99,0xb1,0xe,0xb6,0xa9,0xc7,0x7d,0xb7,0xc9,0xe8,0xc0,0xad,0x44,0x37,0xae,0x20,0xc0,0x6f,0x2,0xe3,0x51,0x76,0x8 } },
{ .uint8 = { 0xfb,0x19,0xaf,0xf9,0x6a,0xf3,0xcf,0xf5,0xe5,0x42,0xbf,0x53,0x56,0x27,0x5,0xd7,0xf4,0xe5,0xd1,0x3a,0x1c,0xf1,0xf3,0x73,0x6f,0xec,0x43,0x27,0x89,0x59,0x74,0xe } },
{ .uint8 = { 0x35,0xf2,0xe6,0xbb,0x6a,0x68,0xdc,0xe2,0x6,0xdf,0x6f,0xe8,0x66,0x73,0xb2,0x38,0x4b,0xb1,0x1b,0x88,0x93,0x7c,0x8f,0x85,0x5,0x74,0x76,0xc0,0xc,0x3,0xce,0x5 } },
{ .uint8 = { 0xaf,0x87,0x9c,0x5e,0xe5,0x3,0xae,0x3f,0xca,0x36,0xba,0x76,0xb2,0x5b,0xb3,0xe2,0xe7,0x9b,0xda,0x59,0x62,0x3d,0x38,0x41,0x0,0x9c,0x63,0x1a,0x48,0xab,0x3,0x5 } },
{ .uint8 = { 0xe,0x70,0x9b,0x40,0x11,0x1e,0xbb,0x67,0x48,0x63,0xc6,0x57,0x1e,0x58,0x2f,0xac,0xa9,0xc7,0x6d,0x66,0xbe,0x23,0x19,0x86,0x10,0x7d,0x80,0x8f,0xe7,0xc1,0x5a,0x12 } },
{ .uint8 = { 0x66,0xcc,0x83,0x4d,0x82,0xf,0x47,0xea,0x6,0xf7,0x42,0x27,0xa2,0x8e,0xf,0x31,0x2b,0x1e,0x9a,0x85,0x4b,0x98,0xfc,0x35,0xcf,0xf4,0x23,0xe7,0x6d,0x62,0xad,0xd } },
{ .uint8 = { 0x8c,0x95,0xff,0x1a,0x60,0xa2,0xba,0xd0,0x78,0xbb,0x57,0x0,0x16,0x97,0xb1,0x2c,0x86,0xed,0x50,0x2b,0xc2,0xa5,0x8c,0xe6,0x5d,0x6,0x56,0xf5,0x37,0xb3,0x78,0x3 } },
{ .uint8 = { 0x68,0xcc,0x3c,0x84,0xc5,0x21,0xf,0xd3,0x34,0xda,0x87,0x4f,0x58,0xd6,0x11,0x11,0x67,0x9e,0x6,0xad,0x28,0xac,0xe9,0xd0,0x9a,0x56,0x91,0x92,0x52,0x4c,0x39,0xe } },
{ .uint8 = { 0x76,0x82,0xfc,0xba,0xf9,0x7c,0x49,0xf7,0x1c,0xfe,0x8c,0xa0,0xf,0x76,0x83,0x4e,0x46,0xa0,0x40,0xcb,0x35,0xa,0x33,0x35,0xce,0x18,0x4f,0xbc,0xa8,0x82,0x3f,0x8 } },
{ .uint8 = { 0xd2,0x16,0x63,0x67,0xab,0x5f,0x8c,0x26,0x33,0x9e,0xa4,0x65,0xad,0x6c,0xb6,0x9d,0x38,0x1,0x51,0xdd,0x74,0xf0,0xb8,0xd9,0x2,0xa3,0x8d,0x16,0x8,0x3,0x34,0x9 } },
{ .uint8 = { 0x47,0x84,0xe9,0x3e,0xbb,0x1c,0x50,0xdd,0x4d,0xe9,0x45,0x79,0x1e,0x28,0x82,0x9e,0x3d,0x4d,0xf7,0xfa,0x49,0xdf,0xf3,0x78,0x6b,0xd0,0x9c,0x57,0x20,0x24,0xcd,0xa } },
{ .uint8 = { 0xf4,0x96,0xad,0xe3,0x8b,0x75,0x1c,0x7a,0xd0,0xb8,0x8d,0x9d,0xa6,0xd9,0x49,0xa1,0xca,0xcf,0x67,0x19,0xcf,0x74,0x60,0xff,0xcb,0x4e,0xf8,0x47,0x7d,0x4e,0x80,0x1 } },
{ .uint8 = { 0xdb,0x5b,0xd,0xae,0xde,0x7f,0x6a,0xa3,0xb2,0xa5,0x53,0x9a,0xd3,0x8c,0xee,0x5a,0xb1,0x2,0x11,0xed,0x50,0xb3,0x79,0x18,0x8d,0x19,0x8b,0xa8,0x4d,0x84,0x93,0xf } },
{ .uint8 = { 0xa2,0xe2,0x79,0x3f,0xa7,0x67,0x95,0xea,0x63,0x65,0xd8,0xd5,0x60,0x33,0x1,0xf8,0xdf,0xf8,0x9a,0xf6,0x2c,0xc7,0x40,0x38,0x31,0x18,0x7,0x42,0x3f,0xcb,0xc2,0x1 } },
{ .uint8 = { 0xb0,0xe3,0x37,0x49,0x6c,0x2b,0x40,0xb,0x88,0x42,0x71,0x37,0xc5,0xcd,0x4d,0x50,0x9c,0xd,0xcc,0x4f,0xf1,0xe2,0xa7,0x42,0x90,0xce,0x91,0x2d,0x67,0xae,0x52,0xf } },
{ .uint8 = { 0xe7,0xf1,0x26,0x6a,0xa8,0xf,0x9c,0xbf,0xab,0xfc,0xbf,0xdb,0x3,0xa5,0x32,0xd,0xd2,0x5e,0x2e,0xbd,0xb4,0x9a,0x1d,0x97,0xed,0x51,0x29,0x42,0x8d,0x8c,0x79,0x8 } },
{ .uint8 = { 0x4c,0xdf,0x75,0xe6,0xe7,0x28,0xb3,0xc0,0x4a,0xde,0x7,0xbe,0xf0,0x2c,0xaa,0x3c,0x5f,0x40,0x96,0xc8,0x22,0x31,0x26,0xc,0xa0,0xc5,0x49,0x12,0x50,0x43,0x72,0xd } },
{ .uint8 = { 0xfe,0x19,0x4f,0x6a,0xed,0xb2,0xc9,0x2e,0x50,0x1f,0x79,0x8,0xa8,0x22,0x75,0x81,0x4f,0x72,0xf5,0xad,0x5a,0x12,0xb1,0x1f,0x8b,0x7a,0xbe,0x7d,0xcd,0xfe,0x6f,0x8 } },
{ .uint8 = { 0xe1,0x31,0xd2,0xeb,0x6c,0xc3,0x48,0x52,0x4b,0x73,0x75,0x93,0x2,0xc5,0x34,0x2b,0xaa,0xc3,0x7,0x8b,0x9c,0xb2,0x65,0xee,0x66,0xc5,0x94,0x68,0x4b,0x1c,0x20,0x0 } },
{ .uint8 = { 0x0,0x86,0xcb,0xa3,0x26,0x1b,0x32,0x5a,0x94,0xf6,0xe1,0x74,0xd1,0xdb,0xe3,0xe7,0x7d,0xda,0xae,0x1c,0xfa,0xe8,0x69,0x24,0x97,0x2d,0x50,0x40,0x96,0x8,0x61,0x3 } },
{ .uint8 = { 0xf1,0xdf,0x11,0x68,0x42,0xc5,0xc0,0xc9,0x33,0xfc,0xaf,0x51,0xc6,0x74,0x39,0xd2,0x40,0xa9,0x3,0x2,0x2c,0x15,0xbe,0xe,0xbc,0xa9,0x51,0x1,0xd0,0x6,0xf5,0x8 } },
{ .uint8 = { 0x1c,0x6f,0xf6,0xef,0x3b,0x40,0x60,0x42,0x14,0x3d,0xf7,0xc9,0x25,0xfb,0xd6,0xc5,0x44,0xc4,0x87,0x2e,0x2f,0x54,0x9d,0x40,0xfe,0x2a,0x36,0x9b,0xe8,0xae,0xd,0x2 } },
{ .uint8 = { 0x40,0x8f,0x2c,0x69,0x4c,0x39,0x71,0xaf,0xae,0xb2,0x62,0xbb,0x66,0x7c,0xbb,0xa,0x41,0x11,0x28,0x16,0xc6,0xcf,0x40,0x3e,0x5e,0xd6,0x6e,0xb4,0xfb,0xdb,0xb1,0x5 } },
{ .uint8 = { 0x77,0xe4,0x63,0xa4,0xf7,0xd1,0x25,0xf5,0xd9,0x59,0xa4,0x75,0x99,0xbc,0x4a,0x47,0x2a,0x9d,0x9c,0x9b,0xb6,0x56,0x47,0x97,0x44,0x4a,0xc5,0x26,0xc8,0x98,0x42,0x9 } },
{ .uint8 = { 0x47,0xa7,0x14,0x6c,0x9f,0xc4,0x54,0x46,0xe1,0xe5,0xa0,0x21,0x23,0x3d,0x6d,0x2e,0x11,0x1b,0x4b,0x8e,0xb2,0x4f,0xcd,0x49,0x82,0x57,0x68,0x8,0x92,0x73,0x8b,0x4 } },
{ .uint8 = { 0x2e,0x15,0x9c,0x6d,0x38,0x9f,0x5b,0x1a,0x16,0xb6,0x9a,0xc3,0xa6,0x11,0xf2,0xfa,0x53,0xc1,0x78,0x9b,0x2d,0x5d,0x6b,0x6b,0x50,0x7d,0x75,0x9d,0xd7,0xab,0x1d,0xa } },
{ .uint8 = { 0xd0,0x7a,0x7b,0x5,0xc2,0x71,0x78,0x2e,0x12,0x1f,0x4e,0x48,0xa1,0xa2,0x95,0xb2,0xea,0x7f,0x63,0xd8,0x68,0xb4,0x97,0x70,0x96,0x9,0x47,0xba,0x1b,0xe4,0x19,0x4 } },
{ .uint8 = { 0xe1,0x65,0xd,0x17,0x7e,0x73,0xe,0x6a,0x2f,0x57,0xb1,0x48,0xa1,0xd7,0xd3,0x61,0x92,0x19,0x79,0xb0,0x72,0x73,0xf4,0x56,0x87,0x9b,0x44,0x2,0x58,0x79,0x14,0x9 } },
{ .uint8 = { 0xd1,0xbf,0xf1,0x18,0x46,0xef,0x99,0x80,0x6e,0xd8,0xac,0x68,0x1,0xc2,0x5b,0x55,0x9,0x21,0x2b,0xfb,0xd3,0xa3,0x23,0x59,0x28,0xa8,0x5,0x42,0x90,0x56,0xd1,0x9 } },

};

//从raw 值 转换为 椭圆曲线上域的值
__constant__  bigint_u256_t PSD2_MDF[9] = {
{.uint64= {0xd3e1b719627eef9c, 0x944031bef18c05b7, 0x5f93620ee657611, 0x92696314ebee66e, } },
{.uint64= {0x54aea0c191840710, 0x7c6572388b138cc5, 0xe99b1822dc11ec04, 0x10d3c62b2749cd64, } },
{.uint64= {0x5370bc9058093ead, 0x931289d50ba24947, 0x9a5ce8875869e795, 0xd53f93182fa097b, } },
{.uint64= {0x8f4dc124731197d2, 0x2a36bbcd15582c3e, 0xc96c0705c91d3bb0, 0xce47d4a9f5a4deb, } },
{.uint64= {0xd204dd5e85dbef9, 0x2e8997d74794711f, 0x5ab70b99149188b, 0xbdcba2f9a789715, } },
{.uint64= {0xae376ca177578394, 0xbb07742229d10c42, 0x685c9a4cddde050b, 0x7bc47e84d8e154d, } },
{.uint64= {0xbca10ed4ff18a096, 0x1f8a77b246da31e1, 0x16c033b302c66787, 0x60a726dd83142a7, } },
{.uint64= {0x79b1112619f1dbe1, 0xf4daee489fff6d9f, 0x34e6125090a92526, 0xb85b6021fb7ced6, } },
{.uint64= {0x20a59b5ebb6c205b, 0x24fd7ff921f78dd2, 0xe5cf0906ceef128f, 0xa754100a042d8b9, } },
};

__constant__  bigint_u256_t PSD2_ARK[117] = {

{.uint64= {0x9ec464191dff626d, 0xe3afe4fc52de2c3e, 0x55098efb31c5bb8a, 0xf51daa50d9eca73, } },
{.uint64= {0x5d10c94384e955b9, 0xa0ff049c6b09597b, 0x88f1e8263b3c7219, 0x72ebc82c44a6f65, } },
{.uint64= {0x49caf68d25cdb9f9, 0xfe42e5325bd12d75, 0xe3e3cb6b932d45af, 0xf66f5ed24cd0873, } },
{.uint64= {0x53ef1989e533d3ca, 0xa45703bbb56d59c1, 0xba8aedc20c662bf6, 0x5a32d3a8421d9ad, } },
{.uint64= {0x6457a7e4c5678437, 0xba7a6987f4f57a6a, 0xf4815167645f1a57, 0xa5dc656713fb8a0, } },
{.uint64= {0x235b0b1555642db5, 0xcec9b78e44325abc, 0x2c9563f8da1d08d0, 0xa57cccb809b2880, } },
{.uint64= {0x1029d909e6027efe, 0xaf9468e7372d5724, 0x3bc1ed72b652b16f, 0x3ff277709c4d56f, } },
{.uint64= {0x8832930f567fa582, 0xa8f83263a19791a3, 0x50d0d3408b6b04e5, 0x37574ef84e565ae, } },
{.uint64= {0xb35c1903ca017011, 0xc1386268816fd161, 0x1445750501a3d4e7, 0x4be77d725b0dd8f, } },
{.uint64= {0xd294950b1e1e3741, 0x3b334cc79a833de3, 0xff5871e159439902, 0x29925877e112131, } },
{.uint64= {0xb3a38d95e1f97fd5, 0x545cf6fe79bb544f, 0x894e3746765042df, 0x889051459c99d4c, } },
{.uint64= {0x299bc35698f34a2c, 0x711b785113e1400b, 0xaac79823d636106f, 0xf8d542c206a2e9f, } },
{.uint64= {0x9938734953d0983, 0x700ed38f8bc8ea7b, 0x49b257f4f65d2812, 0xd72e7d074c95c5b, } },
{.uint64= {0x97737f650c322d2, 0xd0d7b9b75e137f4f, 0x1f92e2819efe9a5b, 0x10a6d950e906c671, } },
{.uint64= {0x1aeb383a27e36474, 0x500e50d8c9564698, 0x870bc0e7677c4113, 0x116378547e3583a9, } },
{.uint64= {0x3f6b9f1ead24181e, 0xaa25f4a586c604a, 0xf8501dd7c7261c88, 0xb6995183189f583, } },
{.uint64= {0xeaeb5379b4aa7a8d, 0x6646f3cc489c24ed, 0x6b27d15587e9a0fc, 0x94100315653d99e, } },
{.uint64= {0x991568444e180311, 0xd2737438c26ce90a, 0xdbc275ba4f952e3d, 0xcf2901c8061a86c, } },
{.uint64= {0x420fe6c269716a0b, 0x8b7dd4708ef709e, 0x329d8a2e6fe200ed, 0x1222e31cd92b62f0, } },
{.uint64= {0xb652e596980e08ad, 0x2a4cc531884fedc8, 0xc46ca74eec39d49d, 0xb48e7b1d7a509fd, } },
{.uint64= {0x4fbaa3701ddeaa0d, 0x46253f48809fe521, 0xd112ddebd5e32030, 0x1097e3ec6e60e118, } },
{.uint64= {0x7f6c938967b0c46, 0x66ad30d9230cadd4, 0x8e29f7672e4a0be7, 0x4c41db88b35e10a, } },
{.uint64= {0x2e06c4c8fccce6bb, 0xcbcc115dfebbaec8, 0xb9bfb46a2bb6c2bd, 0x3f7022fec7006a3, } },
{.uint64= {0x9ba3cd1b5a7ea410, 0x2846943e7e492f26, 0x4b5467dac4c06c22, 0xfee1050a62457c4, } },
{.uint64= {0x596d83e6afb9ad36, 0x691980fd98760a84, 0x1c0fb72236b50683, 0xfb35167160dc2a4, } },
{.uint64= {0x358fb4ef306096a6, 0xb56d4c3b786a0e9e, 0x62bb94eba70f9139, 0x69e36e501704d5f, } },
{.uint64= {0x869458f0325ec1bb, 0x30284fcd2d534c3c, 0xcbef149296fc72ca, 0x3552f3324516b98, } },
{.uint64= {0xb90e6db2001a76e5, 0xb0a48954a7c16964, 0x4c1708dfa8a00722, 0xc282b1347afa86e, } },
{.uint64= {0xc546217837e92b33, 0x16e80e8b0b4c9e40, 0xcbaea3f8561857e8, 0x982959ca9f93cb6, } },
{.uint64= {0x137b20572180ca19, 0x4bec88c627bc3286, 0x437845a137c09c0f, 0xdd87b8180b6642a, } },
{.uint64= {0xef8ae14f80579ec4, 0xff00af70e7f82ca8, 0x349a668ef4a24268, 0x9045a3ec71217dd, } },
{.uint64= {0x575c565182420ff, 0x81855a68599238f4, 0x4e193b16c1e2b053, 0x30c8d8f831a71b9, } },
{.uint64= {0xed1d25ee13da70a5, 0xa803756488855b3e, 0x2d11bea5f5a54c36, 0x64cb52b22e2d297, } },
{.uint64= {0x1f821ae0a7913832, 0x175e966e73ae8438, 0xe28721be2235efa8, 0x116cc59db8e844a2, } },
{.uint64= {0xf53466063aa88798, 0x2b79ed8957bb7285, 0x97b8d49913859afe, 0x947b6de25ca7063, } },
{.uint64= {0x80b1cec4a438e5ca, 0xf9aa695bca686274, 0x65295130b59fbc7a, 0x3454a81643d399, } },
{.uint64= {0x23586842f3dcbd7b, 0xadbbed7a47cd64f6, 0x593a5c48a5ba0b1, 0xe1760cb4dd92ff9, } },
{.uint64= {0xe459ec695b72e1cb, 0xd059df313554e8b2, 0xe797a63fe1b23a52, 0x839931866d47ce4, } },
{.uint64= {0x6ea750cf756662d9, 0x79d1a39f46657aa8, 0x61746109e6df99d, 0xf409eea34d3c04b, } },
{.uint64= {0xe2b164f19764e6cb, 0xe6bbad23d87191ab, 0xc695fe80e0895bfa, 0x3a97f7c1c702554, } },
{.uint64= {0xec79f5a78833fb19, 0x91c7690074f5edb, 0xa0746c7f5da5ea7f, 0x36afe094bfd4997, } },
{.uint64= {0x8535b06e643956d0, 0x71e46f2cb9deda7c, 0x2b8551524c6b29c6, 0x115df1e317d6ab8b, } },
{.uint64= {0x81ddb164cc2b4916, 0xd63ce6f3cbd46b92, 0x69788d568278b866, 0x5b29eb4dc7bfc1e, } },
{.uint64= {0xf96b859445f2c615, 0xa97ca470c11bdb4a, 0x7ce851d297375458, 0xa2198edf79ecc3b, } },
{.uint64= {0x5e33cc209fd478ac, 0x14a3d98d7598304e, 0xb1d9c34490abbcc8, 0x9c86ed5b899a089, } },
{.uint64= {0x6982c91b12222cc3, 0x94b68f470f5beb94, 0x9f8c30285f5c05dc, 0xa05981c0a98b710, } },
{.uint64= {0x2153fb9b52f9a5cc, 0xed6e5f7dd4033f2, 0xd07544ce5b5cd24b, 0x3be7474bc150d13, } },
{.uint64= {0x9a56a64d048f50de, 0x286b69ead08aed25, 0xb41a061755e2d61a, 0x38458139d19b6fc, } },
{.uint64= {0xb3f42fcd95791a2b, 0x9d79b2900c749c2a, 0x13ea88cdc661cad2, 0x51f6f4a81dac2d6, } },
{.uint64= {0x1017d43acc69eb34, 0x409a0366fdfb255c, 0x6dbcb7cba12fd028, 0x6ca7b023199e459, } },
{.uint64= {0xd8f156db12d243b7, 0xb4218274948f02e4, 0x2d93f5aefc12c2a9, 0x5395869efcac3fc, } },
{.uint64= {0x44abcef2bb378bf6, 0x19d41c39a3ac77b2, 0x91c2afbf2fdcc886, 0xe458d57f71ffad6, } },
{.uint64= {0xc642bde093f2dc2, 0x2156fa7445a2704f, 0xc55a1d0705344d4e, 0xc6150424cc81af4, } },
{.uint64= {0xf06d469b327ae6d0, 0x69c61dfacf3e40a2, 0xbae3ec026f6c6672, 0xff33bbef179f8f, } },
{.uint64= {0x862aeb8da2a507c1, 0xeda0be75dda9289e, 0x1729577c9b5a9827, 0xea511b4272c347b, } },
{.uint64= {0xff4ae063f9472246, 0xc4297d2def661cad, 0x385a933723a0e760, 0xc65a9a8a43949cd, } },
{.uint64= {0x9d03995b7b1b474e, 0x36d5a5ad0f8d6072, 0x651d898ee87b7135, 0xb0e118897c3b45, } },
{.uint64= {0xcae938200e46105a, 0x1f6b2efc7872cc03, 0xfbe6168061e31f80, 0xf2272719835b3c5, } },
{.uint64= {0xbe06e0ac481e787a, 0x4e5320ef47f4e988, 0xeb9fc868d7f38492, 0xc73d223c95e12fd, } },
{.uint64= {0xec6695b677cf0913, 0xa7b157ea72ff71b, 0x32fd977ba15f343c, 0x31fd0756e79a9ac, } },
{.uint64= {0x7426205ccae3b157, 0x856d6be2bc010f64, 0x391fb5e9a31f9f38, 0x589644dd9371cc0, } },
{.uint64= {0xc9f7135f2d78a54a, 0xeb5788f04a55ad34, 0x267622b89f984a02, 0x6ab1f5a030e99e1, } },
{.uint64= {0x500c67b669c4765b, 0x4da6fa1b807a9198, 0x4cca1e31ed54966f, 0x9dc9480c4b47a1f, } },
{.uint64= {0x5b9aa111f3d61034, 0x3e6b17d174b4b22d, 0xf964603ba7e1bb93, 0x128614f1a579c017, } },
{.uint64= {0xbe80b8945b441e43, 0x7714c42ee8a722ea, 0x83cb858b82e71e94, 0x11bb67615d302201, } },
{.uint64= {0x2cfa9008d32c72f9, 0xdb89b27958ae6d83, 0x5acd39c9f2940e49, 0xd657c9194233125, } },
{.uint64= {0x7b176b7feba59989, 0x6d8ee1a0b09e5bb3, 0xc134d0e7c43176f0, 0x6db3edfb4c260a, } },
{.uint64= {0x5f2eb47cdfbff1b9, 0x9d95b48e6d2f6793, 0xd09a9ae4b6aabc84, 0xd165c0f7e611b25, } },
{.uint64= {0xed1d02c9e159c1af, 0xcb9e58ab4a9763e7, 0xdde6dddf8faa0e65, 0xb5b1eaa9ac18b7c, } },
{.uint64= {0x771e186a3d6ad630, 0x78fac17292a67062, 0x78902142b5104a6b, 0xfd613359f241b94, } },
{.uint64= {0xf4f326f9505a0e0f, 0xdccb5076a7459e0f, 0x190973660cdd8c05, 0x242310fd779c3b, } },
{.uint64= {0xb874fe8f171a17b4, 0x6df42b9a3709bd13, 0xd38327ceb98797c, 0x10eef1a001cce1f8, } },
{.uint64= {0x1c05d7265d7da8a9, 0x58b260fa34434b12, 0xe808027528326c4f, 0x101e692d3e838010, } },
{.uint64= {0x1a1661b5232dc811, 0x919bd6452d7c1682, 0xa4fb971c6bf20c6f, 0xc6d4311b009b244, } },
{.uint64= {0x4d9254fcd7bb93da, 0x4dd6c35e99e37982, 0x9b0963aca9ead932, 0x681390062fa94ea, } },
{.uint64= {0x5e059d100f1b3f34, 0x23e9c67978c601f7, 0xccb310d0e7a4018d, 0x623d4ba2971122f, } },
{.uint64= {0x116a2078cac7a34b, 0xd94c0edf8476f4b6, 0x132162f3d5aeb5f2, 0x12901bbc2a27c164, } },
{.uint64= {0x5c271c72349a01b8, 0x262d47c9fe072023, 0x49138c4989636720, 0xefb0a0e8b74bcbe, } },
{.uint64= {0xf6b05e568158d1d8, 0xbb8cb0ef8700f134, 0x18d6cedd3931f532, 0xea0b608f8c8a715, } },
{.uint64= {0x114a58e627776dce, 0xfadfac8e64f4283, 0xb40caeea84db18b4, 0xf135db0d8e08986, } },
{.uint64= {0x88345f4a15c0ec0e, 0xecaa89ec71a68724, 0x16a85d9b1f3ce58b, 0x5f144646764db35, } },
{.uint64= {0xba2b234bb6a8d7ad, 0x47d996f15572ca34, 0x69f77b3a09be26c9, 0xb1dba6322b682d1, } },
{.uint64= {0xa3107d5ae4d19ba4, 0x381baa613fa56d49, 0x881af809dd72bcae, 0x4123c3627c60841, } },
{.uint64= {0x2d6135db280dcbc1, 0x95330c86044ea2f9, 0x864b0b1afc615a88, 0x86117f7ab8b4ca2, } },
{.uint64= {0x7696662f8aaf709e, 0x4d420c5ea6ce1644, 0xf31105c111066ad, 0xf2580713531d85b, } },
{.uint64= {0x9e379d8ff43bc547, 0x964d11f183103fab, 0x44a53be274dbe1ad, 0x6a267ad2b2033ab, } },
{.uint64= {0x7a37ea31754d349b, 0xdf63eece1b8badbf, 0xf4f8ab96617e05f3, 0xcb906610127516b, } },
{.uint64= {0x318688a65d1a53d3, 0x7feb26c13ec1630b, 0xae82f204469a8e86, 0xdd6b64a1858126e, } },
{.uint64= {0x66703c46f98c950e, 0x168e49b4ed701705, 0x26962547178879ea, 0xde52afe5395b717, } },
{.uint64= {0x8365113fed764f0c, 0x590e29c31c68e8cc, 0x8a43c53398da1a9d, 0x10ba00dedcc99067, } },
{.uint64= {0x4e88bfec6902e5d7, 0xbb39701f486adf29, 0xf34de7b5a4cd494c, 0x9336475128a6cda, } },
{.uint64= {0xe73bea7689bd3eb2, 0x56c62b3060b33d07, 0x18594b3864cb7166, 0xc047cf731278da0, } },
{.uint64= {0x8f19b50d5b1926e8, 0x929da129a3366b1e, 0x4234f5d772f8cd58, 0xda188ed55431b4d, } },
{.uint64= {0xfff038246a912103, 0x7b8f4bede1c99953, 0x87f1c8f0b548d761, 0xcc0d16bb8d06f11, } },
{.uint64= {0xcc5611d257410762, 0xcfb9252a3290bc01, 0x2b555f4bd9767b58, 0x6ebe3ac3f172e88, } },
{.uint64= {0x2e5d5dfd8d86565f, 0xe503aeda26517138, 0xfe018fe276e0e33d, 0x9f5e383c9a32953, } },
{.uint64= {0x3edab5f0f8de7ed8, 0xb028ef019a59c73e, 0x2d592d52e2da549a, 0xc59ef1f23f0b132, } },
{.uint64= {0x68f8ba6ff8693fa2, 0xa73cfb4799873150, 0xde60a3170e57994d, 0x4b26beea79585be, } },
{.uint64= {0x21fae7cd88c1ea99, 0xf2ed6694e80a29c6, 0x55f427ee15533d45, 0x4091b4623cce678, } },
{.uint64= {0x2168e590097d7fdc, 0x3de49bdf74a09d6a, 0x1d807d5083ac4ffb, 0x114a99fa392f1035, } },
{.uint64= {0x5e763211a68f7bfd, 0x757efb09f06d1fbe, 0xd3ed4aac99b32364, 0x1179f44e98dbfa2d, } },
{.uint64= {0xeb3c03e28ff48138, 0xa404f1c754096f96, 0x422f9c9b0ffbd49f, 0x2d28553ce8d773, } },
{.uint64= {0xc229d9ecfe01ac19, 0xf06795be03d6ae9d, 0x4cfad8f1e6ee49f4, 0xfa1bb36fa18432, } },
{.uint64= {0xdf59b5601c3cd323, 0xdb4ad0ec6d2e9a63, 0x7afb96d36b7c8dfb, 0x23ffae15409e6c0, } },
{.uint64= {0xafd9b28917086ac6, 0xed627b6585aef0bf, 0x1f1b4afdf4a133e3, 0x21b7329bc02e25b, } },
{.uint64= {0x96c3de5813aec901, 0xd191b509038ccb27, 0xc6d4105f483d3884, 0x105773b2d1ff2110, } },
{.uint64= {0x4b6abfe837876890, 0x2b0c259d7a9c5e18, 0xfabae1eed4089bea, 0x118181aca8f07377, } },
{.uint64= {0xaa54445143377422, 0x646da8d02b8b2a43, 0xcc3088e3c70d1170, 0x54812e23a7fbdcc, } },
{.uint64= {0x74f6197a0e79f261, 0xc538d0d224c893fc, 0x353ca81bc5669525, 0xa317eecbaa836d1, } },
{.uint64= {0xe128dfac01da21f8, 0x34e658f0aaa917ba, 0x20737042db392ee4, 0x12008b969d02ab26, } },
{.uint64= {0x228968205643dcf5, 0x16ac507dc2b4810c, 0x37b6529b3c3b881b, 0x5d8cbac60dd80a8, } },
{.uint64= {0xb17db0a3dc14521d, 0x4377c2c7a159716c, 0x881f3e12c9c4c71e, 0xa3bbadeeaadceec, } },
{.uint64= {0x4e6ed3a8acdbf36f, 0x3b35ef84bdbc6fa5, 0xafd736f19bf4ff74, 0x5e6b12e1d90a59d, } },
{.uint64= {0x45b9cd311568ffce, 0x516cb8d6c7992c27, 0xdd9d0e19c9b3a0ec, 0x8677eeb6551763c, } },
{.uint64= {0xfac7e21f9fbf6c01, 0xc1e2cff3ef351175, 0xa4b3ab9e619d711f, 0x1262af5fa841fed9, } },
{.uint64= {0x811bce6f814483fe, 0xbcadf20ebdfd74d, 0xe4d53e15214f5e7f, 0x513673d5124c924, } },
{.uint64= {0x6938ab32caf0e7bc, 0xf0c30598c1145300, 0x1033e8c7338e9131, 0x1e7d4225297a076, } },

};