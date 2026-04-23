# Handoff Plan — Address Co-Author Comments on R&R Round 1 Response Letter

Read `CLAUDE.md` first for project-wide context, then `reports/CLAUDE.md` for the response-letter editing rules (they are mandatory).

## Context

Co-author reviewed [reports/R&R round 1/0_response.tex](reports/R&R round 1/0_response.tex) and returned 18 comments. Some target only the `\reply{}` wording; most require an edit in [reports/main.tex](reports/main.tex) that then propagates to the response letter's `\changes{}` block plus the regenerated `3_diff_DONT_TOUCH.tex`.

**Files in scope:**
- [reports/main.tex](reports/main.tex) — manuscript source (Overleaf-synced)
- [reports/R&R round 1/0_response.tex](reports/R&R round 1/0_response.tex) — reply letter the co-author reviewed
- [reports/R&R round 1/2_new_DONT_TOUCH.tex](reports/R&R round 1/2_new_DONT_TOUCH.tex) — kept in sync with `main.tex`, regenerates the diff
- [reports/R&R round 1/3_diff_DONT_TOUCH.tex](reports/R&R round 1/3_diff_DONT_TOUCH.tex) — regenerated after edits; `\changes{}` snippets pulled from it
- [reports/references.bib](reports/references.bib) — needs one new entry (Pucher & Buehler 2008)

**Rules (from [reports/CLAUDE.md](reports/CLAUDE.md)):**
- `cd reports && git pull` before editing any `.tex` there.
- `\changes{}` is pure latexdiff markup; commentary only in `\reply{}`.
- After each batch of `main.tex` edits, run a writing-style-check pass (no filler/hedge words, sparing em-dashes, no AI-typical phrasing).
- After all manuscript edits: `cp reports/main.tex "reports/R&R round 1/2_new_DONT_TOUCH.tex"` → `latexdiff 1_old_DONT_TOUCH.tex 2_new_DONT_TOUCH.tex > 3_diff_DONT_TOUCH.tex` → re-pull `\changes{}` snippets.

## Resolved decisions (from user)

1. **Table col "RP/SP"** → split into two columns: **"Choice model"** (RP/SP/—) and **"Perception scoring"** (Yes/No).
2. **Table col "Safety metric"** → rename to **"Proxy for safety"**.
3. **Dutch cycling culture reference** → add **Pucher & Buehler (2008)** *Making Cycling Irresistible: Lessons from the Netherlands, Denmark and Germany* (Transport Reviews 28(4): 495–528) as a new bib entry.
4. **`\autoref` inside `\changes{}`** → hard-code figure/table numbers (`Fig.~9`, `Table~2`, etc. — pull from the compiled manuscript). Flag the CLAUDE.md deviation in the commit message.

## Comment-by-comment plan

Grouped to let overlapping edits land together. Each item lists (a) the co-author's issue, (b) what changes in `main.tex`, (c) what changes in `0_response.tex`. Drafts below are what the final text should say, not the final diff markup — the `\changes{}` blocks are regenerated from `3_diff_DONT_TOUCH.tex` at the end.

### Group 1 — Broken cross-references (Comments 1, 10)

**C1 — `\autoref{fig:safety-dist-road}` showed as "?":** label exists at [main.tex:258](reports/main.tex#L258) and resolves there. The error is in the **response letter** at [R&R round 1/0_response.tex:381](reports/R&R round 1/0_response.tex#L381) and [R&R round 1/0_response.tex:583](reports/R&R round 1/0_response.tex#L583) — the response `.tex` has no figure with that label. Compile `main.tex` first to pin the actual figure number, then hard-code `Fig.~<N>` in both `\changes{}` blocks.

**C10 — `\autoref{tab:literature_comparison}` error at [R&R round 1/0_response.tex:606](reports/R&R round 1/0_response.tex#L606):** the response letter's own comparison table is labelled `tab:literature_comparison_response` at [R&R round 1/0_response.tex:450](reports/R&R round 1/0_response.tex#L450), so the other label has no target. Replace both instances of `\autoref{tab:literature_comparison}` in that `\changes{}` block with `Table~<N>` from the rendered manuscript (or "the literature comparison table" where phrasing fits).

### Group 2 — Literature-comparison table (Comments 5, 6, 8, 9, 11)

**C5 — split "RP/SP" into two columns:** replace with **"Choice model"** (RP/SP/—) and **"Perception scoring"** (Yes/No). Categorisation:
- Broach 2012 — RP / No
- Kuiper 2021 — RP / No
- Huber 2024 — RP / No
- Uijtdewilligen 2024 — SP / No
- Yu 2024 — — / Yes
- Ye 2024 — — / Yes
- Lu 2025 — RP / No (uses built-environment features, not a perception score)
- **Our study** — SP / Yes

**C6 — rename "Safety metric" → "Proxy for safety".** Values unchanged (Infrastructure type, Crash + incidents, CV safety score, etc.).

**C8 — push back on novelty (softer, more precise):** current `\reply{}` at [R&R round 1/0_response.tex:475](reports/R&R round 1/0_response.tex#L475) concedes "similar methodological pipelines". Rewrite to explicitly distinguish:
- Yu 2024, Ye 2024: CV → perception score maps; no choice model, no behavioural parameters.
- Lu 2025: CV → built-environment features inside a route-choice model; RP data, no explicit perception score, no WTP.
- Our study: CV-derived perceived-safety score as a continuous variable inside a discrete choice model estimated on SP data, with WTP in WTP space and systematic heterogeneity across 17 demographic dimensions.

**C9, C11 — Section 2.3 revision and jagged `\changes{}`:** the `\changes{}` at [R&R round 1/0_response.tex:606](reports/R&R round 1/0_response.tex#L606) reads as broken mid-sentence. After regenerating the diff, replace with a clean snippet — if too long, split into `\changes{}` + `\changesclean{}`.

**Manuscript edits** (for C5, C6, C8, C9, C11): update the table at [main.tex:196-217](reports/main.tex#L196-L217) (column headers + values), and re-check that the Section 2.3 prose at [main.tex:194](reports/main.tex#L194) names the concrete distinction ("no existing study combines SP + continuous CV-derived safety score + WTP-space estimation") rather than "similar methodological pipelines".

### Group 3 — "Discrete choice modelling framework", not "stated preference framework" (Comment 7)

**C7:** SP is a data-collection design; discrete choice modelling is the framework. Edit in `main.tex`:
- [main.tex:501](reports/main.tex#L501) (Conclusion, first sentence): "…into a stated preference framework with willingness-to-pay estimation and systematic heterogeneity analysis" → "…to explain cycling route choice behaviour within a discrete choice modelling framework, enabling willingness-to-pay estimation and systematic heterogeneity analysis."
- Abstract — same phrase; grep for `stated preference framework` and fix every occurrence.

In the response letter, the `\reply{}` for Comment 3.1 ([R&R round 1/0_response.tex:445](reports/R&R round 1/0_response.tex#L445)) and 3.2 ([R&R round 1/0_response.tex:475](reports/R&R round 1/0_response.tex#L475)) also say "stated preference framework" — rewrite to "discrete choice modelling framework (on stated preference data)".

### Group 4 — Discussion and conclusion wording (Comments 2, 3, 4, 15, 16, 17)

**C2 — more age references:** current paragraph at [main.tex:475](reports/main.tex#L475) cites Kuiper, Uijtdewilligen, De Freitas, Keppner. Add two already in [reports/references.bib](reports/references.bib):
- `rossetti_how_2023` — latent-class segmentation by self-assessed health, which co-varies with age.
- `sener_analysis_2009` — classic demographic segmentation of cycling preferences.

Keep the addition to one clause; no literature dump. No `\reply{}` change beyond naming the added citations.

**C3 — flag household/bills interpretation as speculative:** edit [main.tex:481](reports/main.tex#L481) to open with an explicit speculative frame ("We offer two tentative mechanisms, which our cross-sectional design cannot confirm."), and soften "may have more flexible daily schedules" → "might have more flexible schedules", "Financial security may reduce the pressure" → "Financial security could plausibly reduce the pressure". Mirror the hedging in the `\reply{}` at [R&R round 1/0_response.tex:418](reports/R&R round 1/0_response.tex#L418).

**C4 — sharpen WTP framing at [main.tex:469](reports/main.tex#L469):** state that the WTP magnitude alone does not identify whether cyclists place high utility on safety or low disutility on travel time. Suggested rewrite of the second half of the paragraph:
> "The WTP estimate reflects a relative preference: we cannot tell from the ratio alone whether cyclists place a high utility on safety or a low disutility on travel time. Either way, the accepted additional travel time (≈66 s per safety unit) makes the time spent cycling on a safer route feel worthwhile rather than wasted."

Update `\reply{}` at [R&R round 1/0_response.tex:425](reports/R&R round 1/0_response.tex#L425) to flag this ambiguity.

**C15 — third practical application looks missing in the response:** the manuscript at [main.tex:505](reports/main.tex#L505) lists **three** applications. The response-letter `\changes{}` blocks for this content are split across Comment 3.10 ([R&R round 1/0_response.tex:627](reports/R&R round 1/0_response.tex#L627)), Comment 4.1 ([R&R round 1/0_response.tex:772](reports/R&R round 1/0_response.tex#L772)), and Comment 4.2 ([R&R round 1/0_response.tex:781](reports/R&R round 1/0_response.tex#L781)) — the Comment 4.1 snippet only shows "First" and "Second", which is why the co-author thought one is missing. Fix by making the Comment 4.1 `\changes{}` self-contained (show all three items), and add a cross-pointer in the `\reply{}` where needed.

**C16 — add Roos (Terra) reference for scenario planning:** `terra_understanding_2024` is already in the bib (it's Roos Terra's thesis, CV-enriched DCM with safety implicit). Edit the scenario-planning sentence at [main.tex:505](reports/main.tex#L505): "…enabling scenario planning for infrastructure that does not yet exist, extending the cv-dcm framework of \citet{terra_understanding_2024} to explicit perceived-safety evaluation." Mirror in `\reply{}` at [R&R round 1/0_response.tex:779](reports/R&R round 1/0_response.tex#L779).

**C17 — cost variable defense:** the `\reply{}` at [R&R round 1/0_response.tex:807](reports/R&R round 1/0_response.tex#L807) should acknowledge the attribute set was inherited from Terra's experiment, not chosen by us. Add: "We also note that the choice-experiment attributes (travel time and traffic lights, without monetary cost) were set by the original experiment design of \citet{terra_understanding_2024}; this paper works with that attribute set." Nudge [main.tex:317](reports/main.tex#L317) to the same framing (one sentence).

### Group 5 — Cross-cultural context (Comments 12, 14)

**C12 — Dutch cycling culture:** add Pucher & Buehler (2008) to [reports/references.bib](reports/references.bib):
```bibtex
@article{pucher_making_2008,
  title   = {Making Cycling Irresistible: Lessons from the Netherlands, Denmark and Germany},
  author  = {Pucher, John and Buehler, Ralph},
  journal = {Transport Reviews},
  volume  = {28},
  number  = {4},
  pages   = {495--528},
  year    = {2008},
  doi     = {10.1080/01441640701806612}
}
```
Edit `\reply{}` at [R&R round 1/0_response.tex:683](reports/R&R round 1/0_response.tex#L683) and the cycling-attitudes paragraph at [main.tex:477](reports/main.tex#L477) to add: "All respondents are Dutch, and cycling is a mass-practised everyday skill typically acquired in childhood in the Netherlands \citep{pucher_making_2008}, so the range of cycling proficiency within our sample is probably compressed relative to countries with lower cycling penetration."

**C14 — NL vs DE nuance:** limitation paragraph at [main.tex:497](reports/main.tex#L497) currently lumps NL and DE as "countries with mature cycling infrastructure and high cycling rates". Rewrite: "Both countries have developed cycling infrastructure, though they differ in modal share and infrastructure style — the Netherlands has much higher cycling penetration and more fully separated networks, while Germany leans more on mixed-traffic provision." Cite `pucher_making_2008` here too.

### Group 6 — Perceived safety vs observed behaviour (Comment 13)

**C13:** text at [main.tex:178](reports/main.tex#L178) concludes "…SP methods may be better suited to capture safety preferences latent in observed behavior." Overstated. Rewrite: "…which may reflect either that SP designs surface safety preferences suppressed under the constraints of observed route choice, or that RP safety indicators (self-reported incidents) and CV-derived perception scores measure different constructs. The relative reliability of the two approaches for recovering safety preferences remains open." Mirror in `\reply{}` at [R&R round 1/0_response.tex:692](reports/R&R round 1/0_response.tex#L692) and in the discussion paragraph at [main.tex:465](reports/main.tex#L465).

### Group 7 — Multicollinearity rewrite (Comment 18)

**C18:** paragraph at [main.tex:433](reports/main.tex#L433) frames stepwise selection as "guarding against" multicollinearity, implicitly conceding the reviewer's premise (which is wrong — moderate correlations are a standard-error issue, not identification). Rewrite:
> "Some segmentation features are correlated with each other (e.g., vegetation and terrain tend to co-occur; car and road proportions are positively related). Such correlations are common when predictors describe the same scene and, as \autoref{fig:correlation-matrix} shows, none of the retained features exceeds the conventional threshold (|r|>0.9) at which collinearity impedes parameter identification. Moderate correlations inflate standard errors but do not bias coefficient estimates, and attributes are not required to be independent in a discrete choice model. The stepwise selection we use on top of this is motivated by model parsimony rather than by a multicollinearity concern, and the train-test validation (\autoref{tab:train_test_comparison}) confirms that the retained features generalise to new data."

Update `\reply{}` at [R&R round 1/0_response.tex:826](reports/R&R round 1/0_response.tex#L826) — lead with "Moderate correlations among attributes inflate standard errors but do not invalidate the model"; drop the "stepwise selection guards against this" framing.

## Execution order

1. `cd reports && git pull` (sync Overleaf).
2. Add the Pucher & Buehler 2008 bib entry to [reports/references.bib](reports/references.bib).
3. Apply all `main.tex` edits (Groups 1–7). After each group, run the writing-style-check pass (file-level).
4. Fix the `\autoref` errors in [R&R round 1/0_response.tex](reports/R&R round 1/0_response.tex) (Group 1) — independent of the manuscript.
5. Update every affected `\reply{}` block with the co-author's framing (speculative hedging, Roos pointer, SP-vs-RP nuance, multicollinearity re-framing, cost-design inheritance, discrete-choice-modelling terminology, more age references).
6. `cp reports/main.tex "reports/R&R round 1/2_new_DONT_TOUCH.tex"`.
7. `cd "reports/R&R round 1/" && latexdiff 1_old_DONT_TOUCH.tex 2_new_DONT_TOUCH.tex > 3_diff_DONT_TOUCH.tex`.
8. Repull every affected `\changes{}` block from the regenerated diff, enforcing the CLAUDE.md formatting rules (no commentary, pure latexdiff markup, no ellipsis truncation).
9. Per-comment verification — dispatch one agent per comment (CLAUDE.md QA rule) to check: `\changes{}` matches diff, `\reply{}` addresses the full comment, citations use `\citet{}`/`\citep{}`.
10. Compile:
    - `latexmk -pdf reports/main.tex` → no `?` for any `\autoref`.
    - Compile `0_response.tex` → no `?` in PDF.
11. Commit + push from `reports/`, then pull in Overleaf.

## Verification checklist

- [ ] All 18 co-author comments have a corresponding edit (or a deliberate "no change" noted).
- [ ] `main.tex` compiles without undefined references (grep the `.log`).
- [ ] `0_response.tex` compiles without `?` in PDF.
- [ ] Literature comparison table (manuscript + response-letter copy) has the new columns — "Choice model", "Perception scoring", "Proxy for safety" — with values consistent across the two copies.
- [ ] Pucher & Buehler 2008 bib entry present and cited in two places (Dutch cycling culture claim, NL/DE nuance).
- [ ] "Stated preference framework" replaced with "discrete choice modelling framework" in manuscript + response-letter `\reply{}` blocks.
- [ ] Multicollinearity paragraph in Section 4.2 reframed.
- [ ] `3_diff_DONT_TOUCH.tex` regenerated; `\changes{}` snippets in response letter match.
- [ ] Writing-style-check pass clean on all edited sections.
- [ ] Memory updated with any durable facts (e.g., Pucher & Buehler bib key added to the project).

## Reference: where each comment is anchored

| # | Fragment | Response letter | Manuscript |
|---|---|---|---|
| 1 | Rotterdam main-road finding | [R&R round 1/0_response.tex:381](reports/R&R round 1/0_response.tex#L381), [583](reports/R&R round 1/0_response.tex#L583) | [main.tex:497](reports/main.tex#L497) |
| 2 | Age + Kuiper reference | [R&R round 1/0_response.tex:411](reports/R&R round 1/0_response.tex#L411), [706](reports/R&R round 1/0_response.tex#L706) | [main.tex:475](reports/main.tex#L475) |
| 3 | Household/bills speculative framing | [R&R round 1/0_response.tex:418](reports/R&R round 1/0_response.tex#L418), [420](reports/R&R round 1/0_response.tex#L420) | [main.tex:481](reports/main.tex#L481) |
| 4 | WTP framing "wasting time" | [R&R round 1/0_response.tex:425](reports/R&R round 1/0_response.tex#L425), [429](reports/R&R round 1/0_response.tex#L429) | [main.tex:469](reports/main.tex#L469) |
| 5 | Table column "RP/SP" | [R&R round 1/0_response.tex:454](reports/R&R round 1/0_response.tex#L454) | [main.tex:203](reports/main.tex#L203) |
| 6 | Table column "Safety metric" | [R&R round 1/0_response.tex:454](reports/R&R round 1/0_response.tex#L454) | [main.tex:203](reports/main.tex#L203) |
| 7 | "Stated preference framework" | [R&R round 1/0_response.tex:470](reports/R&R round 1/0_response.tex#L470) | [main.tex:501](reports/main.tex#L501), abstract |
| 8 | Novelty pushback | [R&R round 1/0_response.tex:475](reports/R&R round 1/0_response.tex#L475) | [main.tex:194](reports/main.tex#L194) |
| 9 | Section 2.3 precision | [R&R round 1/0_response.tex:475](reports/R&R round 1/0_response.tex#L475), [477](reports/R&R round 1/0_response.tex#L477) | [main.tex:194](reports/main.tex#L194) |
| 10 | `\autoref{tab:lit…}` error | [R&R round 1/0_response.tex:606](reports/R&R round 1/0_response.tex#L606) | — |
| 11 | Incomplete literature `\changes{}` | [R&R round 1/0_response.tex:606](reports/R&R round 1/0_response.tex#L606) | [main.tex:194](reports/main.tex#L194) |
| 12 | Dutch cycling culture | [R&R round 1/0_response.tex:683](reports/R&R round 1/0_response.tex#L683), [685](reports/R&R round 1/0_response.tex#L685) | [main.tex:477](reports/main.tex#L477) |
| 13 | SP vs RP conclusion | [R&R round 1/0_response.tex:692](reports/R&R round 1/0_response.tex#L692) | [main.tex:178](reports/main.tex#L178), [465](reports/main.tex#L465) |
| 14 | NL/DE nuance | [R&R round 1/0_response.tex:750](reports/R&R round 1/0_response.tex#L750) | [main.tex:497](reports/main.tex#L497) |
| 15 | Third practical application | [R&R round 1/0_response.tex:772](reports/R&R round 1/0_response.tex#L772) | [main.tex:505](reports/main.tex#L505) |
| 16 | Roos scenario-planning pointer | [R&R round 1/0_response.tex:779](reports/R&R round 1/0_response.tex#L779), [781](reports/R&R round 1/0_response.tex#L781) | [main.tex:505](reports/main.tex#L505) |
| 17 | Cost-variable inherited from Terra | [R&R round 1/0_response.tex:807](reports/R&R round 1/0_response.tex#L807), [809](reports/R&R round 1/0_response.tex#L809) | [main.tex:317](reports/main.tex#L317) |
| 18 | Multicollinearity reframing | [R&R round 1/0_response.tex:826](reports/R&R round 1/0_response.tex#L826), [828](reports/R&R round 1/0_response.tex#L828) | [main.tex:433](reports/main.tex#L433) |

## Parallel work (previously tracked in this file)

Model reruns for the RID-63 drop (11,190 rows / 746 individuals / 606 train / 140 test) are in progress on a separate track. As of this writing, 8/17 demographic interaction groups are complete in `reports/models/interaction/safety_demographics_20260420_182818/`. The auto-retry wrapper (`scripts/run_demographics_with_retry.sh`) can resume from that checkpoint. Manuscript `\input{...}` repointing and sample-size number updates (752→746, 11,289→11,190) are still outstanding and must be coordinated with the co-author-comment edits above when touching `main.tex`.
