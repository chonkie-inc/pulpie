use std::io::BufRead;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let jsonl_path = args.get(1).expect("Usage: bench <pages.jsonl>");

    let file = std::fs::File::open(jsonl_path).expect("Cannot open file");
    let reader = std::io::BufReader::new(file);

    let mut pages: Vec<String> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(lang) = val.get("meta").and_then(|m| m.get("language")).and_then(|l| l.as_str()) {
                if lang != "en" {
                    continue;
                }
            }
            if let Some(html) = val.get("html").and_then(|h| h.as_str()) {
                pages.push(html.to_string());
            }
        }
        if pages.len() >= 500 {
            break;
        }
    }

    eprintln!("Loaded {} pages", pages.len());

    // Warmup
    for page in pages.iter().take(5) {
        let _ = hummingbird::extract(page);
    }

    // Profiled benchmark — measure each stage
    let mut t_sanitize: u128 = 0;
    let mut t_parse: u128 = 0;
    let mut t_prune: u128 = 0;
    let mut t_segment: u128 = 0;
    let mut t_classify: u128 = 0;
    let mut t_markdown: u128 = 0;
    let mut t_postprocess: u128 = 0;
    let mut ok = 0usize;
    let mut fail = 0usize;

    let start = Instant::now();

    for page in &pages {
        let t0 = Instant::now();
        let sanitized = hummingbird::clean::sanitize(page);
        t_sanitize += t0.elapsed().as_micros();

        let t0 = Instant::now();
        let mut document = scraper::Html::parse_document(&sanitized);
        t_parse += t0.elapsed().as_micros();

        let t0 = Instant::now();
        hummingbird::clean::prune_boilerplate(&mut document);
        t_prune += t0.elapsed().as_micros();

        let t0 = Instant::now();
        let blocks = hummingbird::segment::segment(&document);
        t_segment += t0.elapsed().as_micros();

        if blocks.is_empty() {
            fail += 1;
            continue;
        }

        let t0 = Instant::now();
        let content_blocks = hummingbird::classify::filter_content(blocks);
        t_classify += t0.elapsed().as_micros();

        if content_blocks.is_empty() {
            fail += 1;
            continue;
        }

        let t0 = Instant::now();
        let mut parts: Vec<String> = Vec::new();
        for block in &content_blocks {
            let md = hummingbird::markdown::md_from(block.element);
            let trimmed = md.trim().to_string();
            if !trimmed.is_empty() {
                parts.push(trimmed);
            }
        }
        let combined = parts.join("\n\n");
        t_markdown += t0.elapsed().as_micros();

        let t0 = Instant::now();
        let result = hummingbird::postprocess::postprocess(&combined);
        t_postprocess += t0.elapsed().as_micros();

        if result.is_empty() {
            fail += 1;
        } else {
            ok += 1;
        }
    }

    let total = start.elapsed();
    let n = pages.len() as f64;

    eprintln!("\n{:=<60}", "");
    eprintln!("PIPELINE PROFILE ({} pages, {} ok, {} fail)", pages.len(), ok, fail);
    eprintln!("{:=<60}", "");
    eprintln!("  {:20} {:>10} {:>10} {:>6}", "Stage", "Total (ms)", "Avg (ms)", "%");
    eprintln!("  {:20} {:>10} {:>10} {:>6}", "-----", "----------", "--------", "---");

    let total_us = t_sanitize + t_parse + t_prune + t_segment + t_classify + t_markdown + t_postprocess;
    let stages: Vec<(&str, u128)> = vec![
        ("sanitize (regex)", t_sanitize),
        ("parse_document", t_parse),
        ("prune_boilerplate", t_prune),
        ("segment (features)", t_segment),
        ("classify (GBM)", t_classify),
        ("markdown", t_markdown),
        ("postprocess", t_postprocess),
    ];

    for (name, us) in &stages {
        let ms = *us as f64 / 1000.0;
        let avg = ms / n;
        let pct = *us as f64 / total_us as f64 * 100.0;
        eprintln!("  {:20} {:>10.1} {:>10.2} {:>5.1}%", name, ms, avg, pct);
    }

    eprintln!("  {:20} {:>10.1} {:>10.2} {:>5.1}%", "TOTAL (measured)", total_us as f64 / 1000.0, total_us as f64 / 1000.0 / n, 100.0);
    eprintln!("  {:20} {:>10.1} {:>10.2}", "TOTAL (wall)", total.as_millis() as f64, total.as_millis() as f64 / n);
    eprintln!("\n  Throughput: {:.1} pages/sec", n / total.as_secs_f64());

    // Now test with different tree counts
    eprintln!("\n{:=<60}", "");
    eprintln!("TREE COUNT vs THROUGHPUT (classify stage only)");
    eprintln!("{:=<60}", "");
    eprintln!("  {:>6} {:>12} {:>12} {:>10}", "Trees", "Time (ms)", "Avg (ms)", "Pages/sec");

    for n_trees in [500, 1000, 2000, 3000, 5000, 7696] {
        let start = Instant::now();
        let mut fast_ok = 0usize;
        for page in &pages {
            let sanitized = hummingbird::clean::sanitize(page);
            let mut document = scraper::Html::parse_document(&sanitized);
            hummingbird::clean::prune_boilerplate(&mut document);
            let blocks = hummingbird::segment::segment(&document);
            if blocks.is_empty() {
                continue;
            }
            let content = hummingbird::classify::filter_content_fast(blocks, n_trees);
            if !content.is_empty() {
                fast_ok += 1;
            }
        }
        let elapsed = start.elapsed();
        let throughput = pages.len() as f64 / elapsed.as_secs_f64();
        eprintln!("  {:>6} {:>12.1} {:>12.2} {:>10.1}  (ok={})",
            n_trees, elapsed.as_millis(), elapsed.as_millis() as f64 / n, throughput, fast_ok);
    }
}
