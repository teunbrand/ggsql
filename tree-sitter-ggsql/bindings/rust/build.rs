use std::path::PathBuf;
use std::process::Command;

fn main() {
    let check = Command::new("tree-sitter").arg("--version").output();
    match check {
        Ok(output) if output.status.success() => {}
        _ => {
            println!("tree-sitter-cli not found. Attempting to install...");
            let installation = Command::new("npm")
                .args(["install", "-g", "tree-sitter-cli"])
                .status()
                .expect("Failed to install tree-sitter-cli");

            if !installation.success() {
                eprintln!("Failed to install tree-sitter-cli");
                std::process::exit(1)
            }
        }
    }

    let regenerate = Command::new("tree-sitter")
        .arg("generate")
        .status()
        .expect("Failed to regenerate tree sitter grammar.");

    if !regenerate.success() {
        eprintln!("Failed to regenerate tree sitter grammar.");
    }

    let dir: PathBuf = ["src"].iter().collect();

    cc::Build::new()
        .include(&dir)
        .file(dir.join("parser.c"))
        .compile("tree-sitter-ggsql");
}
