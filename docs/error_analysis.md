# Pulpie Error Analysis — WebMainBench (English, h2t-canonicalized)

Date: 2026-04-09
Sample: 500 English pages, html2text canonicalized output
Score: P=0.825, R=0.837, F1=0.818

## Benchmark Context

| Method | ROUGE-5 F1 (en) | Notes |
|--------|-----------------|-------|
| Pulpie (h2t) | 0.806 (full 6647) | GBM classifier, html2text canon |
| Pulpie (native md) | 0.745 (full 6647) | Custom markdown converter |
| Dripper 0.6B (paper) | 0.878 (full 7809) | Includes non-English |
| magic-html (paper) | 0.714 (full 7809) | Includes non-English |

Gap to Dripper: ~7pp. Formatting accounts for ~3pp. Content extraction quality accounts for ~4pp.

## F1 Distribution

```
  0.9-1.0:  300  (60.0%)  ########################
  0.8-0.9:   55  (11.0%)  ####
  0.6-0.8:   63  (12.6%)  #####
  0.4-0.6:   31  ( 6.2%)  ##
  0.2-0.4:   20  ( 4.0%)  #
  0.0-0.2:   17  ( 3.4%)  #
      0.0:   14  ( 2.8%)  #
```

71% of pages score >= 0.8. The tail (29%) drags the average down.

## Error Categories

| Category | Count | % | Avg F1 | Description |
|----------|-------|---|--------|-------------|
| Good (F1>=0.8) | 355 | 71.0% | 0.961 | Working correctly |
| Low precision | 36 | 7.2% | 0.522 | Boilerplate leaking into output (pred 5-25x longer than ref) |
| Formatting | 26 | 5.2% | 0.601 | Content correct but markdown format mismatches |
| Moderate | 30 | 6.0% | 0.624 | Mixed precision/recall issues |
| Low recall | 21 | 4.2% | 0.557 | Missing content (only 8-50% found) |
| Both low | 21 | 4.2% | 0.137 | Completely wrong extraction |
| Empty | 11 | 2.2% | 0.000 | No output at all |

## By Difficulty Level

| Level | Count | F1 | P | R | Empty |
|-------|-------|-----|---|---|-------|
| Simple | 144 | 0.893 | 0.894 | 0.908 | 1 |
| Mid | 193 | 0.825 | 0.839 | 0.849 | 4 |
| Hard | 163 | 0.742 | 0.748 | 0.761 | 6 |

## Error Details

### 1. Boilerplate Leaking (7.2% of pages, ~2pp F1 cost)

The GBM classifier keeps blocks it shouldn't. Worst cases have prediction 10-25x longer than reference.

Examples:
- esaral.com: P=0.044, pred=30K vs ref=1.3K — entire sidebar/navigation kept
- inktechnologies.com: P=0.045, pred=6K vs ref=378 — product page cruft
- automobilemag.com: P=0.035, pred=6.9K vs ref=535 — gas price tables

Pattern: small main content surrounded by lots of structured boilerplate (product specs, navigation menus, data tables). The classifier lacks signal to distinguish content tables from navigation tables.

### 2. Formatting Mismatch (5.2% of pages, ~3pp F1 cost overall)

Content extraction is correct but markdown formatting differs from html2text's output. Stripping formatting recovers up to 93pp on individual pages.

Worst formatting gaps:
- ljive.com: F1=0.051, stripped F1=0.984 — table-heavy page, completely different table rendering
- racetecresults.com: F1=0.025, stripped F1=0.866 — race results table
- muslimpro.com: F1=0.122, stripped F1=0.934 — prayer times table
- literotica.com: F1=0.346, stripped F1=1.000 — member page with structured data

Pattern: pages with prominent tables or structured data. html2text renders tables differently from our converter.

### 3. Missing Content / Low Recall (4.2% of pages, ~1pp F1 cost)

Pulpie finds the right area but only extracts a fraction of the content.

Examples:
- squarespace forum: P=1.0, R=0.081 — only first post, missing all replies
- lucianne.com: P=1.0, R=0.081 — article headline only, missing body
- naruto-kun.com: P=1.0, R=0.136 — Q&A page, only first answer

Pattern: forums and multi-section pages where content spans many blocks but the classifier only keeps the first few. Also pages where content is in non-standard containers (not p/h/li tags).

### 4. Complete Failures / Both Low (4.2% of pages)

Pages where pulpie extracts completely wrong content.

Examples:
- moodys.com: dynamic JS-rendered content, no server-side HTML
- taylorpictures.net: image gallery with no text blocks
- megamitensei.fandom.com: wiki sidebar extracted instead of article

### 5. Empty Extractions (2.2% of pages)

Pulpie produces no output at all.

Examples:
- scienceforums.net: forum content in non-block-level elements
- completefrance.com: forum with dynamic loading
- nishamadhulika.com: comment section loaded via JS
- kickass.to: user page with no content blocks

Pattern: forums, dynamic pages, JS-rendered content, non-standard DOM structure without block-level elements.

## Improvement Priorities

Ordered by estimated F1 impact:

1. **Fix boilerplate leaking** (~2pp) — 7.2% of pages
   - Tighten classifier threshold
   - Post-classification filter: if total pred length > 5x median, re-score with higher threshold
   - Add features: ratio of pred to total page text, block position relative to content cluster

2. **Fix formatting** (~3pp) — 5.2% of pages
   - Table rendering is the main gap (html2text vs our converter)
   - Since we use html2text canonicalization for scoring, this gap disappears in that eval path
   - For native markdown: improve table formatting to match html2text conventions

3. **Improve recall on forums/multi-section pages** (~1pp) — 4.2% of pages
   - Forum detection: if page has repeated similar blocks (posts), keep all of them
   - Lower classifier threshold when blocks form a content cluster

4. **Handle empty pages** (~0.5pp) — 2.2% of pages
   - Fallback segmenter for pages with no block-level elements
   - Text node extraction from leaf divs

5. **Fix complete failures** (~0.5pp) — 4.2% of pages
   - JS-rendered pages: nothing we can do without a browser
   - Image galleries: detect and return empty gracefully
   - Wiki pages: fix sidebar vs article confusion
