# ICM — Remove stale BWMA avian / MSM veg nav tabs

**Date:** 2026-09-30 (PT)  
**Repo:** `flooded-acreage` (`inyo-gov/flooded-acreage`)  
**Site:** https://inyo-gov.github.io/flooded-acreage/

---

## Goal

Drop **BWMA avian** and **MSM veg** from the production navbar. Those tabs were out of date / went nowhere useful on the flooded-acreage site. Keep **Dashboard** + **About** (+ GitHub icon).

Screenshot context: live navbar on https://inyo-gov.github.io/flooded-acreage/ still showed the two extra tabs after `_quarto.yml` already listed only Dashboard + About.

## Decision

| Item | Behavior |
|------|----------|
| Navbar | Dashboard · About · GitHub only |
| BWMA avian / MSM veg | Removed from nav (not linked from this site chrome) |
| Source of truth | `_quarto.yml` `website.navbar.left` (already correct; rebuilt HTML) |

## Changes

1. **`_quarto.yml`** — unchanged vs HEAD (navbar already Dashboard + About). Local WIP that re-added the tabs was discarded.
2. **`docs/index.html`** — re-rendered with `quarto render`. Removed stale nav links to `bwma-avian` / `bwma-msm-veg`. Also cleared an accidental About-page block that had been appended into `index.html` on a prior render (companion-product body links went with it).
3. **`docs/about.html`** — already matched desired nav; re-render left it clean.

## Live confirmation

After push: https://inyo-gov.github.io/flooded-acreage/ navbar should show only Dashboard + About (+ GitHub).

## Out of scope

- Did not change sibling repos `bwma-avian` / `bwma-msm-veg`.
- Did not add body-text companion links on About (those remain off production for this site).
- Unrelated local lidar ICM docs / `subunits.geojson` WIP left uncommitted.

## Follow-up

- If a program hub link is wanted later, prefer a single “Related” item or README note — not dead nav tabs.
