use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    // Only set up the memory layout linker search path for embedded targets
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("thumbv") {
        let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        fs::copy("memory.x", out.join("memory.x")).expect("failed to copy memory.x");
        println!("cargo:rustc-link-search={}", out.display());
    }
    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rerun-if-changed=build.rs");
}
