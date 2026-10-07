# Lab 21 — Evaluation Report

**Họ tên**: Nguyen Ngoc Han  **MSSV**: 2A202602511  **Ngày**: 2026-10-07
**Tier (đo thật)**: `CPU`  **Base model (đo thật)**: `Qwen/Qwen3.5-0.8B`  **GPU thực tế**: `không có (Apple arm64, CPU-only)`
**Tier dự kiến nộp**: `T4` (`unsloth/Qwen3.5-4B`, Colab Free T4) — NB2–NB5 chưa chạy, xem §3/§4/§5.

> Mọi con số dưới đây khớp với file trong `results/` ở commit này.
> Chạy đo thật: `COMPUTE_TIER=CPU .venv/bin/python notebooks/01_data_and_mask.py`
> (`.env` hiện ghi `COMPUTE_TIER=T4`, nhưng máy này không có GPU nên NB1 được đo ở tier CPU).
> Test suite: `116 passed, 3 skipped` (`requirements-cpu.txt`).

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định, `data/train_seed.jsonl` = 250 dòng, không đổi corpus) |
| Train / val | `225` / `25` (seed 42, `data/split/train.jsonl` + `val.jsonl` do NB1 sinh) |
| `max_length` | `512` (tier CPU) — p95 đo được là `98`, `suggested_max_length = 256` *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | `2` (mặc định `.env`, chưa train — áp dụng cho NB3 khi có GPU) |

**Lựa chọn + lý do.** Giữ corpus và model mặc định để làm quen pipeline trước (đúng gợi ý trong README:
chạy một lượt mặc định trước khi đổi dataset). Tier CPU + `Qwen/Qwen3.5-0.8B` là lựa chọn duy nhất
khả thi trên máy không GPU — và cũng là tier lab thiết kế cho đúng việc này: NB1 + toàn bộ test.

**Về `max_length`.** p95 = 98, max = 101, tier đặt 512. NB1 in cảnh báo: p95 gợi ý 256 nhưng tier đặt 512.
Tôi giữ 512 theo tier (không sửa config giữa chừng) và ghi nhận ở đây: với corpus này, 512 vượt xa nhu cầu
(~5× p95), tốn activation/KV-cache vô ích nếu train ở tier này. Khi chạy tier T4 (`max_length=1024`),
cần đo lại p95 trên tokenizer 4B rồi quyết — nguyên tắc rubric 1.3 là số phải từ đo, không đoán.

**Template có giữ khối `<think>` không?** **Có** — *(results/template_check.json)*:
`verdict = "reasoning preserved — safe to train on traces"`. Probe `thinking_survives()` render
`2+2?` kèm trace 2 bước và thu lại đầy đủ, nên train trên trace là an toàn ở tier này.

---

## 2. Mask proof (NB1 — đã đo thật trên CPU)

| | |
|---|---|
| `supervised_fraction` | `0.3936` (37/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss *(results/mask_proof.json → `supervised_preview`)*:

```
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần bị mask (prompt + scaffold) không vào loss:

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

**Ý nghĩa.** Cả hai assert xanh: loss chỉ tính trên câu trả lời JSON + `<|im_end|>` (giữ stop signal
được giám sát), đạt `supervised_fraction ≈ 0.39` — xa ngưỡng `≥ 0.95` là dấu hiệu tính loss cả lên prompt
(rubric 1.1). Tokenizer thật (`Qwen/Qwen3.5-0.8B`) tái hiện đúng bài học F-01: so token-list quanh biên
`<think>` có thể vỡ do token hoá khác nhau ở newline; vì vậy mask suy từ offsets trên text đã render.

---

## 3. Ba baseline (NB2 — CHƯA CHẠY, cần GPU)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | *chưa đo* | *chưa đo* | *chưa đo* | *chưa đo* |
| (b) base + optimized prompt | *chưa đo* | *chưa đo* | *chưa đo* | *chưa đo* |
| (c) LoRA fine-tune | *chưa đo* | *chưa đo* | *chưa đo* | *chưa đo* |

**Trạng thái thật.** `results/` hiện chỉ có `mask_proof.json`, `template_check.json`, `token_stats.json`
(NB1). Chưa có `results/baselines_frozen.json` vì NB2 cần GPU — mốc **chưa được đóng băng**, và theo quy tắc
lab, mốc phải đóng băng **trước** khi train. Tôi không bịa số vào bảng này.

**Kế hoạch chạy (Colab Free T4, tier T4 mặc định):**

```bash
cp .env.example .env   # COMPUTE_TIER=T4, EPOCHS=2, EVAL_LIMIT unset (bản nộp phải để mặc định)
make nb2               # đóng băng eval + đo (a), (b) trước khi train — kỳ vọng ~17–23 ph
```

**(b) có thật sự mạnh hơn (a) không?** Chưa đo được. Kỳ vọng từ số đo đã công bố trong repo
(`docs/MEASURED-T4-2026-08-20.md`, `SIMULATION-FINDINGS.md`): (b) đạt target ~0.76–0.77 với format 1.000
và nhanh ~3× so với (a) (prompt tối ưu dặn chỉ trả JSON nên model dừng sớm thay vì lan man tới token cap).
Tôi sẽ xác nhận lại trên run của mình; nếu (b) không hơn (a), tôi cải thiện (b) theo hướng **mạnh lên**
(siêu chặt schema + ví dụ) và khai báo đúng như rubric yêu cầu — tuyệt đối không làm yếu (b) đi
(`make verify` kiểm tra SHA prompt, làm yếu là lỗi liêm chính).

---

## 4. Giải phẫu cấu hình sai (NB4 — CHƯA CHẠY, cần GPU)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | *chưa train* | 1e-4 | *—* | *—* | *—* | *—* |
| `attn_only` | q,v | *(matched lúc chạy)* | *—* | 1e-4 | *—* | *—* | *—* | *—* |
| `wrong_lr` | text-linear | 16 | *—* | 1e-5 | *—* | *—* | *—* | *—* |
| `qlora` | text-linear | 16 | *—* | 1e-4 | *—* | *—* | *—* | *—* |

Kế hoạch: `make nb3 && make nb4` trên T4 (NB3 ~15–25 ph, NB4 ~45–60 ph). Bốn run chung một ngân sách
step từ `train.planned_steps()` (EPOCHS áp cho cả NB3 lẫn NB4 — cố ý để đối chứng công bằng), `attn_only`
dùng `matched_rank()` khớp ngân sách tham số <5% (`make verify` kiểm tra tự động).

> Xếp hạng bằng cột **target**, không bằng cột train loss (Lỗi #3). Số đo đã công bố trên RTX 3060/0.8B
> (`SIMULATION-FINDINGS.md`) cho thấy vì sao: `attn_only` từng có train loss *thấp hơn* `correct`
> nhưng thua trên target (0.935 vs 0.990) — chấm bằng loss sẽ cho kết luận ngược.

**4.1 — vị trí vs rank:** *chưa có số của run mình.* Dự đoán theo deck §10.2 + số đo công bố: `correct`
(text-linear, r=16) thắng `attn_only` (q,v, rank khớp ngân sách ~r=271–283) trên target, dù hai bên cùng
ngân sách tham số — tức **vị trí gắn adapter là đòn bẩy, không phải rank**. Sẽ xác nhận/khẳng định lại
bằng cột target ở NB5 §4.

**4.2 — `wrong_lr`:** *chưa có số của run mình.* Dự đoán: LR thang full-FT (1e-5, thấp hơn 10×) cho loss
giảm chậm/dẹt, train loss cao rõ rệt (~0.09 vs ~0.05 công bố) và target sụp (~0.325 công bố). Nếu chỉ nhìn
loss mà không biết LR, dễ kết luận sai là "cấu hình này học kém vì thiếu capacity" trong khi nguyên nhân
thật là một con số step-size.

**4.3 — `qlora`:** *chưa có số của run mình.* Dự đoán: tiết kiệm ~25–41% VRAM (2.29 vs 3.08 GB trên 0.8B;
7.15 vs 12.07 GB trên T4/4B công bố) với chi phí nhỏ nhưng thật về accuracy (~0.93 vs 0.99 công bố).
Số của tôi sẽ trả lời khuyến nghị "không dùng QLoRA cho dòng model này": nếu gap target nhỏ mà VRAM
là nút thắt (T4 chỉ 14.6 GB khả dụng), QLoRA vẫn là lựa chọn hợp lý cho inference-budget — không phải
tín điều.

---

## 5. Phán quyết (NB5 — CHƯA CHẠY, cần GPU)

**Kết quả cổng hồi quy**: *CHƯA CÓ* (`results/verdict.json` chưa tồn tại)
`target Δ = —` · `regression Δ = —` · `valid_trace_rate = —`

Diễn giải: chưa thể phán quyết. Điều duy nhất đã được chứng minh ở thời điểm này là pipeline **truyền được
học vào eval** ở tầng nguyên lý (mask đúng + template giữ reasoning + corpus đủ 250 mẫu) — chưa phải bằng
chứng fine-tune thắng baseline. Phán quyết PASS/FAIL chỉ có nghĩa sau khi NB2 đóng băng mốc và NB5 chấm đủ
4 nhóm (target · regression · format · latency). Một FAILED được phân tích tốt (ví dụ catastrophic
forgetting như run 0.8B công bố: target +0.495 nhưng regression −0.578) ăn điểm cao hơn PASSED không giải
thích được — tôi sẽ viết ≥100 từ diễn giải khi có số thật. Lệnh chạy: `make nb5` (~21 ph trên T4).

---

## 6. Định tính — CHƯA CÓ (cần NB5, ≥5 ca gồm ≥2 ca FT thua)

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | *—* | *—* | *—* | *—* | (FT thắng — chờ NB5) |
| 2 | *—* | *—* | *—* | *—* | (FT thắng — chờ NB5) |
| 3 | *—* | *—* | *—* | *—* | ❌ **FT thua — chờ NB5** |
| 4 | *—* | *—* | *—* | *—* | ❌ **FT thua — chờ NB5** |
| 5 | *—* | *—* | *—* | *—* | (chờ NB5) |

Cam kết chống cherry-pick (rubric 3.4): chỉ chọn ca thắng = mất trắng mục này. Tôi sẽ lấy từ
`results/qualitative.json` sau NB5, bắt buộc có ≥2 ca FT thua, rồi tìm mẫu chung (giả thuyết sơ bộ từ
F-31: các ca ticket thiếu tên sản phẩm nguyên văn hoặc intent biên `hoi_thong_tin`/`san_pham_loi` dễ thua
vì model phải nội hoá label space mà prompt train rút gọn không còn schema đầy đủ).

---

## 7. Kết luận & điều tôi học được

**Kết luận (tình trạng hiện tại; chưa phải phán quyết deploy cuối cùng).** Tôi hiện **chưa nên deploy**
bản fine-tune nào: chưa train adapter, chưa đóng băng baseline, và chưa có kết quả NB5. Quyết định này
không có nghĩa là fine-tuning chắc chắn thất bại; nó có nghĩa là hiện chưa có bằng chứng để biện minh cho
việc đưa một model đã thay đổi vào sử dụng. Điều đã được xác minh là một điều kiện cần: trên tokenizer thật
của `Qwen/Qwen3.5-0.8B`, mask giám sát 37/94 token (0.3936), chứa câu trả lời và loại câu hỏi khỏi loss.
Điều đó làm giảm rủi ro học lại prompt, nhưng không chứng minh chất lượng phân loại tốt hơn.

Đòn bẩy đầu tiên ở đây là **tính đúng đắn của pipeline**, cụ thể là mask đã đo; đòn bẩy cuối cùng cho chất
lượng vẫn chưa thể xếp hạng giữa vị trí adapter, learning rate, chất lượng dữ liệu và prompt, vì NB3–NB5
chưa chạy. Kết quả công bố trong repo chỉ là tham khảo cho model/tier khác, không thể thay số đo của lần chạy
này. Tương tự, `template_check.json` xác nhận một trace thử nghiệm được giữ lại, nhưng corpus của tôi chứa
ticket → JSON thường và không có trace suy luận; không thể suy từ probe đó rằng quá trình fine-tune sẽ bảo
tồn reasoning. Trước khi huấn luyện, tôi cần chạy NB2 và chấp nhận mốc (a)/(b) đã đóng băng. Sau đó, chỉ
nếu bản `correct` cải thiện target so với baseline (b), giữ regression trong ngưỡng, tạo JSON hợp lệ và có
latency phù hợp thì mới xem xét deploy. Nếu bất kỳ điều kiện nào không đạt, tôi sẽ không deploy và sẽ dùng
phân tích lỗi để chọn bước tiếp theo — chẳng hạn replay data khi regression suy giảm — thay vì nới cổng sau
khi đã thấy kết quả. Vì các artefact đó hiện chưa tồn tại, quyết định deploy cuối cùng vẫn đang chờ NB2–NB5.

**Ba điều tôi học được (cụ thể, từ việc chạy thật NB1 + đọc repo):**
1. Mask phải chứng minh bằng offsets trên text đã render, không bằng niềm tin vào flag library: NB1 đo thật
   37/94 token (0.3936) với answer-in-loss = true, question-masked = true — và TRL `assistant_only_loss`
   trên Qwen3.5 hoặc crash hoặc giám sát mask khác với mask đã chứng minh (F-10).
2. Tokenizer thật có biên token nhập (`\n` vs `\n\n` quanh `<think>`) làm vỡ mọi phép diff token-list —
   đây là lý do `build_example()` render text rồi tokenize một lần với offsets (F-01), tái hiện đúng trên
   `Qwen/Qwen3.5-0.8B` ở run của tôi.
3. p95 → `max_length` không phải số trang trí: p95 = 98 gợi ý 256 trong khi tier đặt 512 — cứ để tier
   phình mà không đo là đốt VRAM/activation vô ích, và trên T4 (14.6 GB khả dụng, checkpoint 9.32 GB)
   biên an toàn rất hẹp.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** chạy `EPOCHS=1 EVAL_LIMIT=8` pipeline trên Colab T4 (~17 ph NB1+NB2+
NB3+NB5) để có phán quyết đầu tiên, rồi mới quyết có đầu tư full-budget không — đúng hai cần gạt README
cho phép khi bó thời gian (nhưng nộp bài thì để mặc định, `make verify` từ chối smoke run).

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap (`make nb6`, cần GPU)
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (mask-only CPU probe done; no trace corpus or GPU eval, `valid_trace_rate` not measured; see `BONUS-CHALLENGE-EN.md`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

---

## Tái lập (reproducibility)

```bash
cp .env.example .env
.venv/bin/pip install -r requirements-cpu.txt
COMPUTE_TIER=CPU .venv/bin/python notebooks/01_data_and_mask.py  # ~1 ph, đã chạy 2026-10-07
.venv/bin/python -m pytest tests/                                 # 116 passed, 3 skipped
.venv/bin/python scripts/verify.py --smoke                       # 7 passed
# Khi có GPU (Colab T4): make pipeline && make verify
```

Artefact hiện có: `results/mask_proof.json`, `results/template_check.json`, `results/token_stats.json`,
`data/split/{train,val}.jsonl`. Còn thiếu cho bài nộp: `baselines_frozen.json`, `adapters/correct/`,
`runs.csv` (4 dòng), `verdict.json`, `autopsy.json`, `qualitative.json`.

**Trạng thái cổng nộp:** `python scripts/verify.py --smoke` đạt 7/7; full `python scripts/verify.py`
hiện chưa qua do thiếu đúng bốn artefact GPU: `results/baselines_frozen.json`, `results/runs.csv`,
`results/verdict.json`, `results/autopsy.json`. Cần chạy NB2 → NB5 trên GPU rồi chạy lại `make verify`;
không được coi smoke gate là sẵn sàng nộp.
