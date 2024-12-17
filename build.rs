
use std::env;
use std::path::PathBuf;

fn main() {
    let build_dir = PathBuf::from("build-testlib");
    let tests_dir = build_dir.join("test_lib");
    let fmt_dir = build_dir.join("fmt");

    // println!("cargo:rustc-link-search=native={}" , c_src_dir.to_string_lossy());
    // println!("cargo:rustc-link-lib=static=aleocuda"); // 链接名为 libfoo.a 的静态库    

    // println!("cargo:rustc-link-search=native={}", lib_dir.to_string_lossy());
    // println!("cargo:rustc-link-lib=static=aleocuda"); 

    println!("cargo:rustc-link-search=native={}", tests_dir.to_string_lossy());
    println!("cargo:rustc-link-lib=static=testlib");

    println!("cargo:rustc-link-search=native={}", fmt_dir.to_string_lossy());
    println!("cargo:rustc-link-lib=static=fmt");

    //TODO: support windows
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64"); // 指定库文件所在的目录
    println!("cargo:rustc-link-lib=cudart"); 
    println!("cargo:rustc-link-lib=stdc++"); 

    return;    

}
