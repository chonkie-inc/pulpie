# Design: `<pulpie/>` banner

**Date:** 2026-06-29
**Branch:** `branding-banner`
**Status:** Approved (pending final spec review)

## Goal

A primary banner/logo for the **pulpie** package, used on the GitHub README, the
PyPI project page, and the introductory blog post. It must establish pulpie's
identity as the *framework* and sit coherently alongside the `<orange>` mandarine
mark used for the *models*.

## Concept

A bold monospace HTML self-closing tag: **`<pulpie/>`**.

- Wordmark only — no tagline, no icon.
- `pulpie` rendered in ink; the `<`, `>`, and `/` rendered in the Orange-family
  signature color.
- The self-closing tag form signals "this library speaks markup" (pulpie consumes
  HTML), and the bracketed-tag shape is the visual thread shared with `<orange>`.

## Brand architecture (package vs models)

| Mark | Represents | Treatment |
|------|-----------|-----------|
| `<pulpie/>` | the framework / package | ink + monospace + code-tag; GitHub-HTML orange only on the `/` |
| `<orange>` | the model family | vivid illustrated mandarine, full orange |

Same "bracketed-tag" family → clear ecosystem link. Different media (ink/mono vs
illustrated/color) → clear tier separation, so the two never blur together. The
single orange `/` accent is the deliberate connector — the self-closing slash is
the most pulpie-specific glyph (markup that closes/processes).

## Specifications

- **Font:** JetBrains Mono **Bold** (SIL OFL — free to embed and rasterize).
  Developer-native, highly legible, neutral "serious tool" character; pairs with
  README code blocks.
- **Wordmark text:** `<pulpie/>`
- **Colors:**
  - `<pulpie`, `>` glyphs (wordmark + brackets): `#1A1A1A` (light) / `#FFFFFF` (dark)
  - `/` accent: `#D64C2B` — the HTML language color as rendered in GitHub's UI. Semantically apt
    (pulpie processes HTML); in the orange family of the `<orange>` model mark.
  - Background: transparent (PNGs), reads correctly on white and dark.
- **Dimensions:** ~1280×320 (4:1), generous horizontal padding; wordmark centered.

## Assets

| File | Purpose | Notes |
|------|---------|-------|
| `assets/banner.svg` | source of truth, blog hero, scalable | vector; text converted to paths so the font need not be installed by viewers |
| `assets/banner-light.png` | PyPI (white-bg only) + GitHub light theme | ink wordmark + brackets, HTML-orange `/` |
| `assets/banner-dark.png` | GitHub dark theme | white wordmark + brackets, HTML-orange `/` |

Asset location: `pulpie/assets/` (kept with the package; referenced via raw GitHub
URLs so PyPI can load them).

## Wiring across surfaces

- **GitHub README:** `<picture>` with `prefers-color-scheme` media queries to swap
  `banner-light.png` / `banner-dark.png`. (Top-level `README.md` and
  `pulpie/README.md` both updated.)
- **PyPI:** renders `pulpie/README.md` but strips `<picture>` and is unreliable
  with SVG, and uses a white background → the README's image must resolve to
  `banner-light.png` via an absolute raw GitHub URL. Use a single `<img>` fallback
  inside/after the `<picture>` so PyPI picks up the light PNG.
- **Blog post:** use `banner.svg` (or a high-res PNG export) as the hero.

## Surface constraints that drove the above

- PyPI: white background only; strips raw HTML/`<picture>`; unreliable SVG →
  canonical asset is a PNG that looks right on white.
- GitHub: light + dark themes; supports `<picture>` theme switching.
- Blog: wide hero; SVG/high-res PNG fine.

## Out of scope

- The `<orange>` model mark itself (already exists; not modified here).
- Favicon / social-card crops, additional logo lockups — can follow later if needed.
- README copy rewrite beyond inserting the banner.

## Verification

- Render `banner.svg` → both PNGs; visually confirm against the approved reference
  (`~/Downloads`): bold mono `<pulpie/>`, orange `< > /`, ink/white wordmark.
- Confirm light PNG reads on white (PyPI) and dark PNG reads on GitHub dark.
- Preview the README `<picture>` block in both GitHub themes; confirm the PyPI
  `<img>` fallback resolves to an absolute raw URL.
