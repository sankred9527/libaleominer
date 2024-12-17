
is.eq r1 r0 into r14;
    is.eq r3 r2 into r15;
    is.eq r5 r4 into r16;
    is.eq r7 r6 into r17;
    is.eq r9 r8 into r18;
    is.eq r11 r10 into r19;

    hash.psd2 r12 into r20 as u8;
    hash.psd2 r13 into r21 as u8;

    hash.psd2 r12 into r22 as u16;
    hash.psd2 r13 into r23 as u16;

    hash.psd2 r12 into r24 as u32;
    hash.psd2 r13 into r25 as u32;

    hash.psd2 r12 into r26 as u64;
    hash.psd2 r13 into r27 as u64;

    hash.psd2 r12 into r28 as u128;
    hash.psd2 r13 into r29 as u128;

    mul.w r3 r2 into r30;
    mul.w r5 r4 into r31;
    mul.w r7 r6 into r32;
    mul.w r9 r8 into r33;
    mul.w r11 r10 into r34;

    ternary r15 r30 r2 into r35;
    ternary r16 r31 r4 into r36;
    ternary r17 r32 r6 into r37;
    ternary r18 r33 r8 into r38;
    ternary r19 r34 r10 into r39;

