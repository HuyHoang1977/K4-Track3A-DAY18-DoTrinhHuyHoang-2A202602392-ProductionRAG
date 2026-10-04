# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** DoTrinhHuyHoang  
**Khóa:** K4 - Track 3A

---

## RAGAS Scores

Run cuối: 20 câu hỏi từ `test_set.json`, embedding `BAAI/bge-m3`,
reranker `BAAI/bge-reranker-v2-m3`, LLM judge `gemini-3.5-flash-lite`.

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8438 | 0.8417 | −0.0021 |
| Answer Relevancy | 0.7956 | 0.8224 | +0.0268 |
| Context Precision | 0.8000 | 0.8333 | +0.0333 |
| Context Recall | 0.8250 | 0.8667 | +0.0417 |

Cả 4 metric của Production đều trên ngưỡng 0.75; 3/4 cải thiện so với
baseline. `faithfulness` chênh −0.0021, coi như hòa.

### Cấu hình Production

| Thành phần | Baseline | Production |
|---|---|---|
| Chunking | paragraph 500 chars (57 chunks) | parent 2048 → child 256, **gộp ≤400 chars (72 đơn vị)** |
| Retrieval | dense only | BM25 + dense, RRF |
| Reranker | không | cross-encoder, top-20 → top-3 |
| Enrichment | không | Gemini (summary + HyQA + context + metadata) |

### Điều chỉnh đã thực hiện dựa trên đo

| Thay đổi | Đo được (proxy recall, 20 câu) |
|---|---|
| child as-is, 125 đơn vị (median 186c) | 0.7310 |
| bỏ child <100c | 0.7374 |
| **gộp child ≤400c, 72 đơn vị (median 297c)** | **0.8416** |
| chỉ nối tiền tố context cho đơn vị ≥200c | +0.0506 vs nối tất cả |

---

## Bottom-5 Failures

Nguồn: `reports/ragas_report.json` → `failures`.

### #1 — score 0.3333
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** (xem `test_set.json`)
- **Got:** (xem log run — pipeline không lưu answer từng câu)
- **Worst metric:** faithfulness = 0.0
- **Metric chi tiết:** faith 0.0 | ans_rel 0.0 | prec 1.0 | recall 0.3333
- **Error Tree:** Output sai → **Context sai** → Query OK → ✗
- **Root cause:** `context_precision = 1.0` nhưng `context_recall = 0.3333`.
  Các chunk được truy hồi đều liên quan (prec tuyệt đối) nhưng thiếu thông tin
  cần thiết. Câu hỏi hỏi về *hai* điều kiện song song (ai phê duyệt + cần gì
  từ CNTT); top-3 chỉ chứa đủ một điều kiện. Đây là lỗi của câu hỏi nhiều điều
  kiện chứ không phải lỗi chunking.
- **Suggested fix:** `context_recall` thấp ở câu hỏi ghép nhiều điều kiện →
  tăng `HYBRID_TOP_K`, hoặc dùng câu hỏi tách đôi trong test set.

### #2 — score 0.375
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Worst metric:** answer_relevancy = 0.0
- **Metric chi tiết:** faith 1.0 | ans_rel 0.0 | prec 0.0 | recall 0.5
- **Error Tree:** Output sai → **Query OK** → Context **thiếu một phần** → ✗
- **Root cause:** Dạng y hệt #1 — hỏi đồng thời về nghỉ phép **và** lương,
  thuộc hai tài liệu khác nhau. `faithfulness = 1.0` nghĩa là LLM **không**
  bịa — nó trả lời đúng với context nó có, chỉ là context thiếu.
- **Suggested fix:** Multi-hop. Cần chunk có thể chứa thông tin từ cả hai nguồn,
  hoặc truy hồi theo từng sub-question rồi hợp nhất.

### #3 — score 0.7066
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Worst metric:** faithfulness = 0.0
- **Metric chi tiết:** faith 0.0 | ans_rel 0.8265 | prec 1.0 | recall 1.0
- **Error Tree:** Output sai → **Context đúng hoàn toàn** → Query OK →
  → **sinh ra từ câu trả lời, không phải retrieval**
- **Root cause:** `precision = 1.0` và `recall = 1.0` — context chứa đúng
  thông tin cần thiết. Nhưng `faithfulness = 0` nghĩa là câu trả lời chứa
  thông tin **không có trong context**. Đây là hallucination thuần của bước
  generate.
- **Suggested fix:** Siết prompt, giảm `temperature` (đang 0.3), hoặc bắt buộc
  trích dẫn nguồn.

### #4 — score 0.7459
- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Worst metric:** faithfulness = 0.0
- **Metric chi tiết:** faith 0.0 | ans_rel 0.9837 | prec 1.0 | recall 1.0
- **Error Tree:** Cùng dạng #3 — **Output sai, Context hoàn hảo** → ✗
- **Root cause:** Retrieval đúng tuyệt đối nhưng LLM vẫn bịa. Câu hỏi này
  có đáp án nằm ở bảng theo giới tính/tình huống; LLM có thể đảo ngược điều
  kiện (ví dụ quy đổi "kết hôn" thành "nghỉ phép năm").
- **Suggested fix:** Prompt phải yêu cầu trích nguyên văn từ context.

### #5 — score 0.7643
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Worst metric:** context_precision = 0.3333
- **Metric chi tiết:** faith 0.8333 | ans_rel 0.8907 | prec 0.3333 | recall 1.0
- **Error Tree:** Output **đúng** → Context **thừa** → ✗ (lỗi nhẹ nhất)
- **Root cause:** `recall = 1.0` (đủ thông tin) nhưng `precision = 0.3333`:
  2/3 chunk trả về là nhiễu. Đây là hệ quả trực tiếp của việc gộp chunk ≤400
  chars — đơn vị lớn chứa nhiều dòng bảng lương không liên quan tới dòng
  Junior.
- **Suggested fix:** Đây là đánh đổi đã biết của V2 (recall +0.11, precision
  −0.05). Giảm `RETRIEVAL_MERGE_MAX` xuống ~300 nếu ưu tiên precision.

---

## Nhận xét tổng quan

### 3/5 failure có `faithfulness = 0` dù context hoàn hảo

#3 và #4 có `context_precision = 1.0` **và** `context_recall = 1.0` — nghĩa là
retrieval hoàn toàn đúng, nhưng LLM vẫn sinh ra thông tin ngoài context.
**Cải thiện retrieval không giúp được nhóm lỗi này; phải sửa ở bước generate.**

Đây là phát hiện quan trọng nhất: nó cho thấy `context_recall` và `faithfulness`
là hai trục độc lập, và việc tối ưu truy hồi đã đi vào ngõ cụt với nhóm lỗi này.

### 2/5 failure là do câu hỏi nhiều điều kiện

#1 và #2 hỏi đồng thời hai thông tin ở hai tài liệu khác nhau. Top-3 chunk
và rerank top-3 không đủ giữ cả hai. Đây là giới hạn của kiến trúc
single-hop hiện tại, không phải lỗi cấu hình.

### Điểm yếu còn lại

| Vấn đề | Số liệu |
|---|---|
| Latency reranker | ~2260ms cho 20 docs trên CPU, vượt xa ngưỡng 150ms |
| PDF scan | 2 file PDF không có text layer bị bỏ qua (`BCTC.pdf`, `Nghi_dinh_so_13`) |
| Độ nhiễu đo | Cùng cấu hình cho kết quả khác nhau giữa các run (~0.05) |

---

## Case Study (cho presentation)

**Question chọn phân tích:**
> *"Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?"* (failure #4)

Vì đây là ca minh hoạ rõ nhất cho kết luận trên: retrieval hoàn hảo,
generation sai.

**Error Tree walkthrough:**

1. **Output đúng?** → ✗ SAI. `faithfulness = 0.0`
2. **Context đúng?** → ✓ ĐÚNG HOÀN TOÀN. `precision = 1.0`, `recall = 1.0`.
   Top-3 chunk chứa chính xác thông tin về nghỉ phép kết hôn.
3. **Query rewrite OK?** → ✓ Không có bước rewrite; truy vồn đã khớp.
4. **Fix ở bước:** → **Bước 4 (generate)**, không phải bước retrieval.

**Bài học:** Với câu hỏi này, toàn bộ 3 module truy hồi (M1 chunking, M2 hybrid
RRF, M3 cross-encoder) đã làm đúng việc. Điểm mất là do prompt generate
cho phép LLM suy diễn thay vì trích dẫn. Sửa prompt + `temperature` thấp hơn
là đủ — **không cần đụng vào chunking hay reranker.**

**Nếu có thêm 1 giờ, sẽ optimize:**
- Sửa prompt generate: bắt buộc trích nguyên văn, cấm thêm thông tin ngoài context
- Giảm `temperature` từ 0.3 xuống 0.1
- Chạy lại ≥3 lần mỗi cấu hình để ước lượng độ nhiễu trước khi kết luận
