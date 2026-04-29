use std::collections::HashMap;
use std::io::BufRead;
use std::time::Instant;

fn ngrams<'a>(tokens: &'a [&'a str], n: usize) -> HashMap<Vec<&'a str>, usize> {
    let mut counts = HashMap::new();
    if tokens.len() >= n {
        for window in tokens.windows(n) {
            *counts.entry(window.to_vec()).or_insert(0usize) += 1;
        }
    }
    counts
}

fn rouge_n_f1(reference: &str, prediction: &str, n: usize) -> f64 {
    let ref_tokens: Vec<&str> = reference.split_whitespace().collect();
    let pred_tokens: Vec<&str> = prediction.split_whitespace().collect();

    if ref_tokens.len() < n || pred_tokens.len() < n {
        return 0.0;
    }

    let ref_ngrams = ngrams(&ref_tokens, n);
    let pred_ngrams = ngrams(&pred_tokens, n);

    let mut overlap: usize = 0;
    for (ng, pred_count) in &pred_ngrams {
        if let Some(ref_count) = ref_ngrams.get(ng) {
            overlap += (*pred_count).min(*ref_count);
        }
    }

    let ref_total: usize = ref_ngrams.values().sum();
    let pred_total: usize = pred_ngrams.values().sum();

    if ref_total == 0 || pred_total == 0 || overlap == 0 {
        return 0.0;
    }

    let precision = overlap as f64 / pred_total as f64;
    let recall = overlap as f64 / ref_total as f64;
    2.0 * precision * recall / (precision + recall)
}

struct Page {
    html: String,
    reference: String,
    level: String,
}

fn run_pipeline_with_trees(page: &Page, n_trees: Option<usize>) -> String {
    let sanitized = hummingbird::clean::sanitize(&page.html);
    let mut document = scraper::Html::parse_document(&sanitized);
    hummingbird::clean::prune_boilerplate(&mut document);

    let blocks = hummingbird::segment::segment(&document);
    if blocks.is_empty() {
        return String::new();
    }

    let content_blocks = match n_trees {
        Some(n) => hummingbird::classify::filter_content_fast(blocks, n),
        None => hummingbird::classify::filter_content(blocks),
    };

    if content_blocks.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();
    for block in &content_blocks {
        let md = hummingbird::markdown::md_from(block.element);
        let trimmed = md.trim().to_string();
        if !trimmed.is_empty() {
            parts.push(trimmed);
        }
    }

    let combined = parts.join("\n\n");
    hummingbird::postprocess::postprocess(&combined)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let jsonl_path = args.get(1).expect("Usage: bench_rouge_trees <pages.jsonl>");

    let file = std::fs::File::open(jsonl_path).expect("Cannot open file");
    let reader = std::io::BufReader::new(file);

    let mut pages: Vec<Page> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
            let lang = val.get("meta")
                .and_then(|m| m.get("language"))
                .and_then(|l| l.as_str())
                .unwrap_or("");
            if lang != "en" {
                continue;
            }

            let html = val.get("html").and_then(|h| h.as_str()).unwrap_or("").to_string();
            let reference = val.get("convert_main_content").and_then(|r| r.as_str()).unwrap_or("").to_string();
            let level = val.get("meta")
                .and_then(|m| m.get("level"))
                .and_then(|l| l.as_str())
                .unwrap_or("unknown")
                .to_string();

            if !html.is_empty() && !reference.is_empty() {
                pages.push(Page { html, reference, level });
            }
        }
    }

    eprintln!("Loaded {} English pages with references", pages.len());

    let tree_counts: Vec<Option<usize>> = vec![
        Some(500), Some(1000), Some(2000), Some(3000), Some(5000), None,
    ];

    eprintln!("\n{:=<80}", "");
    eprintln!("TREE COUNT vs ROUGE-5 F1 (full pipeline, {} pages)", pages.len());
    eprintln!("{:=<80}", "");
    eprintln!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}  {:>6}  {:>6}",
        "Trees", "All", "Simple", "Mid", "Hard", "Pages/sec", "OK", "Empty");
    eprintln!("  {:->6}  {:->8}  {:->8}  {:->8}  {:->8}  {:->10}  {:->6}  {:->6}",
        "", "", "", "", "", "", "", "");

    for n_trees in &tree_counts {
        let label = match n_trees {
            Some(n) => format!("{}", n),
            None => "all".to_string(),
        };

        let start = Instant::now();
        let mut scores_all: Vec<f64> = Vec::new();
        let mut scores_simple: Vec<f64> = Vec::new();
        let mut scores_mid: Vec<f64> = Vec::new();
        let mut scores_hard: Vec<f64> = Vec::new();
        let mut empty = 0usize;
        let mut ok = 0usize;

        for (i, page) in pages.iter().enumerate() {
            let prediction = run_pipeline_with_trees(page, *n_trees);

            if prediction.is_empty() {
                empty += 1;
                scores_all.push(0.0);
                match page.level.as_str() {
                    "simple" => scores_simple.push(0.0),
                    "mid" => scores_mid.push(0.0),
                    "hard" => scores_hard.push(0.0),
                    _ => {}
                }
                continue;
            }

            ok += 1;
            let r5 = rouge_n_f1(&page.reference, &prediction, 5);
            scores_all.push(r5);
            match page.level.as_str() {
                "simple" => scores_simple.push(r5),
                "mid" => scores_mid.push(r5),
                "hard" => scores_hard.push(r5),
                _ => {}
            }

            if (i + 1) % 1000 == 0 {
                let avg = scores_all.iter().sum::<f64>() / scores_all.len() as f64;
                eprint!("\r  trees={}: {}/{} pages, avg R5={:.4}", label, i + 1, pages.len(), avg);
            }
        }

        let elapsed = start.elapsed();
        let throughput = pages.len() as f64 / elapsed.as_secs_f64();

        let avg_all = scores_all.iter().sum::<f64>() / scores_all.len().max(1) as f64;
        let avg_simple = scores_simple.iter().sum::<f64>() / scores_simple.len().max(1) as f64;
        let avg_mid = scores_mid.iter().sum::<f64>() / scores_mid.len().max(1) as f64;
        let avg_hard = scores_hard.iter().sum::<f64>() / scores_hard.len().max(1) as f64;

        eprint!("\r");
        eprintln!("  {:>6}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>10.1}  {:>6}  {:>6}",
            label, avg_all, avg_simple, avg_mid, avg_hard, throughput, ok, empty);
    }

    eprintln!("\n  Reference comparisons (from Dripper paper):");
    eprintln!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}", "Method", "All", "Simple", "Mid", "Hard");
    eprintln!("  {:->6}  {:->8}  {:->8}  {:->8}  {:->8}", "", "", "", "", "");
    eprintln!("  {:>6}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}", "DS-V3", 0.9098, 0.9415, 0.9104, 0.8771);
    eprintln!("  {:>6}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}", "Drip6B", 0.8779, 0.9205, 0.8804, 0.8313);
    eprintln!("  {:>6}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}", "magicH", 0.7138, 0.7857, 0.7121, 0.6434);
    eprintln!("  {:>6}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}", "trafil", 0.6402, 0.7309, 0.6417, 0.5466);
}
