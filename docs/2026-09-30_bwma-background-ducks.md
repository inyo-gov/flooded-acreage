# ICM — Subtle background ducks on BWMA dashboard

**Date:** 2026-09-30 (PT)  
**Repo:** `flooded-acreage` (`inyo-gov/flooded-acreage`)  
**Site:** https://inyo-gov.github.io/flooded-acreage/

---

## Goal

Add a light, non-distracting flock of flying ducks in the page background (BWMA waterfowl theme) without competing with KPI cards or maps.

## Decision

| Item | Behavior |
|------|----------|
| Visual | 5 small SVG silhouettes, ~8–13% opacity, slow L→R flight |
| Interaction | `pointer-events: none`; `aria-hidden`; behind content (`z-index: 0`) |
| Motion | CSS keyframes; **`prefers-reduced-motion: reduce`** → static parked silhouettes (2 hidden) |
| Layout | No change to KPI / map / archive structure; ducks inject into `body` only |

## Changes

1. **`styles.css`** — `#bwma-duck-sky`, `.bwma-duck`, `@keyframes bwma-duck-fly`, reduced-motion rules; raise `.navbar` / `main.content` / footer above the sky layer.
2. **`includes/dashboard-head.html`** — tiny DOMContentLoaded injector (inline SVG flock).
3. **`docs/`** — `quarto render` of `index.qmd` + `about.qmd` for Pages.

## Out of scope

- Did **not** touch season-peak / current-season KPI logic in `index.qmd` (still in flight elsewhere).
- No EE re-runs; no CSV / report HTML changes.

## Live

https://inyo-gov.github.io/flooded-acreage/
