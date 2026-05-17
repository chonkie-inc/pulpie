"""Compare ROUGE with and without markdown formatting."""
import json, subprocess, tempfile, os, re
from collections import Counter

BENCH_PATH = "data/webmainbench.jsonl"
HBIRD_BIN = "target/release/pulpie"

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def rouge5(ref, pred):
    rt = ref.split(); pt = pred.split()
    if not rt or not pt: return 0.
    rn = Counter(ngrams(rt, 5)); pn = Counter(ngrams(pt, 5))
    if not rn or not pn: return 0.
    ov = sum((rn & pn).values())
    p = ov / max(sum(pn.values()), 1)
    r = ov / max(sum(rn.values()), 1)
    if p + r == 0: return 0.
    return 2*p*r/(p+r)

def strip_md(text):
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[*\-]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = text.replace('|', ' ')
    text = re.sub(r'---+', '', text)
    return text

with open(BENCH_PATH) as f:
    lines = f.readlines()

scores_normal = []
scores_stripped = []

for i in range(200):
    rec = json.loads(lines[i])
    html = rec.get("html", "")
    ref = rec.get("convert_main_content", "")
    if not html or not ref:
        continue
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html)
        tmp = f.name
    try:
        r = subprocess.run([HBIRD_BIN, tmp], capture_output=True, text=True, timeout=30)
        pred = r.stdout.strip()
    except:
        pred = ""
    finally:
        os.unlink(tmp)
    if not pred:
        scores_normal.append(0.0)
        scores_stripped.append(0.0)
        continue
    scores_normal.append(rouge5(ref, pred))
    scores_stripped.append(rouge5(strip_md(ref), strip_md(pred)))

n = len(scores_normal)
print(f"200-page comparison:")
print(f"  With MD formatting:    {sum(scores_normal)/n:.4f}")
print(f"  Stripped formatting:   {sum(scores_stripped)/n:.4f}")
diff = (sum(scores_stripped) - sum(scores_normal)) / n
print(f"  Difference:            {diff:+.4f}")
print(f"  (Positive = formatting is still hurting us)")
