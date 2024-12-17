program puzzle.aleo;

function synthesize:

    input r0 as boolean.public;
    input r1 as boolean.public;
    input r2 as i8.public;
    input r3 as i8.public;
    input r4 as i16.public;
    input r5 as i16.public;
    input r6 as i32.public;
    input r7 as i32.public;
    input r8 as i64.public;
    input r9 as i64.public;
    input r10 as i128.public;
    input r11 as i128.public;
    input r12 as field.public;
    input r13 as field.public;




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

    is.eq r39 -170141183460469231731687303715884105728i128 into r40;
    add.w r39 1i128 into r41;
    ternary r40 r41 r39 into r42;
    neg r42 into r43;
    lt r13 r12 into r44;
    square r12 into r45;
    square r13 into r46;
    xor r37 r7 into r47;
    shr.w r23 r22 into r48;
    mul r46 r45 into r49;
    cast.lossy r49 into r50 as i128;
    lte r29 r28 into r51;
    gt r36 r5 into r52;
    sub.w r27 r26 into r53;
    lte r23 r22 into r54;
    xor r50 r39 into r55;
    mul r49 r46 into r56;
    cast.lossy r56 into r57 as i8;
    is.eq r38 -9223372036854775808i64 into r58;
    add.w r38 1i64 into r59;
    ternary r58 r59 r38 into r60;
    neg r60 into r61;
    add.w r25 r24 into r62;
    add.w r36 r5 into r63;
    add.w r29 r28 into r64;
    mul r56 r49 into r65;
    cast.lossy r65 into r66 as i64;
    add.w r64 r29 into r67;
    add.w r66 r38 into r68;
    add.w r53 r27 into r69;
    lt r63 r36 into r70;
    is.eq r55 -170141183460469231731687303715884105728i128 into r71;
    add.w r55 1i128 into r72;
    ternary r71 r72 r55 into r73;
    neg r73 into r74;
    gte r68 r66 into r75;
    gt r63 r36 into r76;
    square r56 into r77;
    add.w r47 r37 into r78;
    mul.w r78 r47 into r79;
    square r65 into r80;
    xor r69 r53 into r81;
    ternary r76 r23 r22 into r82;
    square r77 into r83;
    is.eq r63 -32768i16 into r84;
    add.w r63 1i16 into r85;
    ternary r84 r85 r63 into r86;
    neg r86 into r87;
    lte r67 r64 into r88;
    lte r67 r64 into r89;
    gt r23 r22 into r90;
    shr.w r21 r23 into r91;
    mul r83 r80 into r92;
    cast.lossy r92 into r93 as i16;
    mul r92 r83 into r94;
    cast.lossy r94 into r95 as u8;
    pow.w r67 r62 into r96;
    is.eq r68 -9223372036854775808i64 into r97;
    add.w r68 1i64 into r98;
    ternary r97 r98 r68 into r99;
    neg r99 into r100;
    gte r81 r69 into r101;
    is.eq r92 0field into r102;
    ternary r102 r12 r92 into r103;
    shl.w r79 r95 into r104;
    gte r79 r78 into r105;
    lte r68 r66 into r106;
    lte r81 r69 into r107;
    gte r55 r50 into r108;
    xor r81 r69 into r109;
    lt r67 r64 into r110;
    add r103 r94 into r111;
    gte r79 r78 into r112;
    is.eq r103 0field into r113;
    ternary r113 r12 r103 into r114;
    gte r114 r111 into r115;
    mul r114 r111 into r116;
    square r114 into r117;
    xor r67 r64 into r118;
    gte r93 r63 into r119;
    mul.w r93 r63 into r120;
    gt r79 r78 into r121;
    mul.w r109 r81 into r122;
    not r95 into r123;
    sub.w r79 r78 into r124;
    xor r62 r25 into r125;
    gte r120 r93 into r126;
    lt r122 r109 into r127;
    xor r23 r22 into r128;
    mul r117 r116 into r129;
    is.eq r68 -9223372036854775808i64 into r130;
    add.w r68 1i64 into r131;
    ternary r130 r131 r68 into r132;
    neg r132 into r133;
    gte r118 r67 into r134;
    shl.w r55 r95 into r135;
    gt r128 r23 into r136;
    is.eq r0 r1 into r137;
            

