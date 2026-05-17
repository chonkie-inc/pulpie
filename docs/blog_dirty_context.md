# Dirty Context, Dirty Answers: Why Web Cleaning Matters for AI Agents

## Blog Post Plan

### Hook
- Agents are browsing the web at inference time now — not just training time
- Every major agent framework (OpenAI, Anthropic, Google) has web browsing tools
- The web content going into context windows is shockingly dirty

### The Training Case (brief — already established)
- AICC, FineWeb, DataComp all proved: cleaner data → better models
- Dripper/MinerU-HTML built a whole pipeline for this — 986K pages labeled
- Not controversial — everyone agrees training data should be clean

### The Inference Case (novel — the real argument)
- Agents call Exa, Parallel, Firecrawl, Jina, Browserbase to fetch pages
- What comes back: nav bars, cookie banners, ads, "related articles", footer links, login prompts, schema.org JSON, tracking pixels
- Show concrete before/after examples (2-3 real pages from Exa/Parallel)
- qrater clean rate numbers on API outputs vs Pulpie-cleaned outputs

### Failure Mode 1: Context Pollution
- Context window is finite and expensive
- 30-50% filled with boilerplate = truncated real content
- Agent must reason over noise — like reading a book where every other page is junk mail
- Retrieval-augmented generation becomes retrieval-augmented confusion

### Failure Mode 2: Hallucination from Noise
- Model can't distinguish nav text from page content
- "Related Articles" sidebar → model cites articles that aren't about the topic
- Breadcrumbs like "Home > Products > Enterprise" get interpreted as content
- Structured data (price tables, spec sheets) from sidebars leak into answers

### Failure Mode 3: Ad/CTA Injection
- Model faithfully reproduces promotional content
- "Start your free trial", "Subscribe for $9.99/mo", affiliate links
- Agent answers become ads — user asks about a product, gets a sales pitch
- Trust erosion: users can't tell if the agent is helping or selling

### The Pareto Efficiency Argument
- Show the quality/compute tradeoff curve
- CPU tier: Trafilatura (0.640) → magic-html (0.714) → Pulpie Espresso (0.808)
- GPU tier: Orange Base 0.6B (0.847) → Dripper 0.6B (0.854) → Orange Large 2.1B (0.862)
- For inference pipelines, latency matters — you need cleaning in the hot path
- Pulpie Espresso: best CPU cleaner, 16.8pp above trafilatura
- Pulpie Orange: matches Dripper quality at lower deployment complexity

### What Good Cleaning Looks Like
- Before/after on 3 representative pages (simple blog, complex e-commerce, forum)
- ROUGE-5 and qrater scores for each
- Show the agent's answer with dirty vs clean context

### Call to Action
- If you're building agents that browse the web, you need cleaning in your pipeline
- Not just for training anymore — every API call is a cleaning problem
- Link to Pulpie / Chonkie
