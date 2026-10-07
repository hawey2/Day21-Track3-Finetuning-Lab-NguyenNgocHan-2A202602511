# Bonus Challenges — Lab 21 (English)

> Do these after `make verify` is green. Each one connects to a **new** section of the
> 2026 deck (Part A: pretrain → mid-train → post-train → optimizers → architecture).
> Record results in the Appendix of `submission/REPORT.md`.

> **Run status (2026-10-07):** CPU checks and the B3 mask comparison are recorded below.
> GPU-dependent training/evaluation challenges are not claimed complete: this run has no
> GPU, and `results/` does not contain trained adapters or NB2–NB5 evaluation artefacts.

---

## B1 — Merge & multi-adapter serving (+3) · deck §23

Run `make nb6`. Required: `results/merge_check.json` showing post-merge score does not
drop by more than 0.01, and ≥2 adapters hot-swapped on one loaded base.

**Question:** merging gives zero inference overhead — what do you give up? When would you
keep adapters separate anyway?

---

## B2 — Your own domain dataset (+3) · deck §17

≥200 high-quality examples. Requires `data/CUSTOM_DATASET.md`: source · collection method
· decontamination · **why this data is distributionally new** relative to what the base
model already saw (deck §3.3).

200 careful examples usually beat 2,000 scraped ones — 2026 bases are saturated on
generic web text.

---

## B3 — Reasoning-trace collapse (+4) · deck §17.5 ⭐ hardest

Fine-tuning a reasoning model on ordinary Q→A data **destroys its reasoning while task
accuracy keeps rising**. No familiar metric catches it.

```bash
MASK_MODE=assistant-only make nb3 && make nb5   # record valid_trace_rate
MASK_MODE=response-only  make nb3 && make nb5   # record valid_trace_rate
```

### Local check (CPU; mask only, not the bonus experiment)

On the shipped 250-example corpus and `Qwen/Qwen3.5-0.8B`, the first training example
produced the same mask in both modes: **37/94 supervised tokens (0.3936)**, with identical
decoded supervised text. This agrees with the repository's documented caveat: the shipped
answers are bare JSON and the empty `<think></think>` scaffold is part of the generation
prefix, so there is no reasoning trace in the supervised span for `response-only` to
exclude. NB1's `mask_proof.json` intentionally records the standard `assistant-only` proof;
the direct comparison was made with `labkit.data.build_example()` for both modes.

| MASK_MODE | Mask on first sample | target | **valid_trace_rate** | regression | Status |
|---|---:|---:|---:|---:|---|
| assistant-only | 37/94 (0.3936) | not measured | not measured | not measured | mask checked; no GPU training/eval |
| response-only | 37/94 (0.3936), identical to assistant-only | not measured | not measured | not measured | mask checked; no GPU training/eval |

**Conclusion:** This does not reproduce reasoning-trace collapse. It confirms the two mask
modes are a no-op on this corpus, so comparing their `target` or `valid_trace_rate` after
training would not test the intended hypothesis. No adapter was trained and no NB5 metric
was produced. To complete B3, use a decontaminated custom corpus with actual reasoning
traces inside assistant answers, verify NB1 shows a difference between the masks, then
train and evaluate both runs on GPU with the same base, split, and step budget.

**Question:** did `target` rise while `valid_trace_rate` fell? If you had only looked at
`target`, would you have noticed?

> The direction is **model-dependent** — the source study found empty-think catastrophic
> for Qwen3-8B but protective for Llama-R1-8B. Do not generalize from one model.

---

## B4 — A *controlled* rank sweep (+3) · deck §11

The old lab's centrepiece, done properly: **hold** `target_modules="text-linear"` fixed,
sweep only `r ∈ {8, 16, 64}`, same LR and step budget.

**Question:** rank the three knobs — rank, placement, learning rate — by effect size,
with numbers. Does your 250-sample dataset carry enough information for r=64 to use?

---

## B5 — HuggingFace Hub (+2)

`model.push_to_hub("<user>/lab21-qwen35-triage-vi")`, link in the report.

---

## B6 — Ungraded: optimizer mismatch · deck §7.3

Switching to **Muon** to fine-tune an **Adam-pretrained** model *degrades* quality —
"optimizer mismatch" — with severity proportional to update magnitude, which is why
**LoRA makes it survivable**. Try it. Do not carry the Adam learning rate over.

Write your prediction first. A wrong prediction you can explain beats a lucky guess.

---

## B7 — Ungraded: MoE route-aware LoRA · deck §7.5

On an MoE base, expert routing is heavily skewed, so adapting every expert wastes most of
the adapter. Profile routing counts on a small calibration set, adapt only the **top 25%
routed experts per layer**, compare against full LoRA.

Published result: within ±1pp of full LoRA at 70–73% fewer trainable parameters; random
expert selection at the same budget is ~2.5pp worse — the *routing signal* is what works.
And **do not train the router** — vendors disable it by default for stability.
