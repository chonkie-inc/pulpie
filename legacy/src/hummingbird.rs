use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut html_mode = false;
    let mut file_path = None;

    for arg in &args[1..] {
        if arg == "--html" {
            html_mode = true;
        } else {
            file_path = Some(arg.as_str());
        }
    }

    let html = match file_path {
        Some(path) => std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Error reading {}: {}", path, e);
            std::process::exit(1);
        }),
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).unwrap_or_else(|e| {
                eprintln!("Error reading stdin: {}", e);
                std::process::exit(1);
            });
            buf
        }
    };

    let result = if html_mode {
        pulpie::extract_html(&html)
    } else {
        pulpie::extract(&html)
    };

    match result {
        Ok(output) => print!("{}", output),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
