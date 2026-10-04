# Reflection — Lab 18: Production RAG

**Họ và tên:** DoTrinhHuyHoang  
**Khóa:** K4 - Track 3A

> Bản nháp viết từ diễn biến thật của quá trình làm. Các phần đánh giá chủ quan

## 1. Điều khó nhất: không phải viết code mà là tin vào số đo

Hai lần tôi tin vào bằng chứng sai và phải đính chính.

**Lần 1 — tin proxy thay vì hệ thống thật.** Thử nghiệm chunking cho thấy proxy
recall giảm khi gộp chunk (`context_precision` 0.2348 → 0.1823). Tôi dự đoán
RAGAS cũng giảm. Khi chạy thật, `context_precision` **tăng** 0.8167 → 0.8333.
Proxy đo overlap từ khóa không thấy điều mà LLM judge thấy.

**Lần 2 — cơ chế đúng, kết luận sai.** Tôi phân tích tiền tố context chiếm 89%
nội dung ở chunk 3 ký tự, rồi kết luận nó làm giảm truy hồi. Ablation bác bỏ:
prefix thực ra **giúp** `+0.0840`. Vấn đề không phải có tiền tố, mà là áp tiền tố
cho cả mảnh vụn — bằng chứng đã nằm ngay trong số liệu tôi thu mà không nhận ra.

---

## 2. Ba lỗi kỹ thuật đáng nhớ

| Lỗi | Triệu chứng | Nguyên nhân thật |
|---|---|---|
| `429 RESOURCE_EXHAUSTED` | 81 lần/run | Free tier 15 RPM, code gọi tuần tự không throttle |
| `400 Multiple candidates` | 35 lần, `answer_relevancy = 0.0` | RAGAS gửi `n>1`, Gemini chỉ nhận `n=1` |
| `501 UNIMPLEMENTED` | 40 lần | Tầng tương thích OpenAI của Gemini **không có** `/embeddings` |

**Lỗi 501 nguy hiểm nhất** vì nó không hiện ra: nó biến `answer_relevancy` thành
`NaN`, rồi code quy đổi thành `0.0` — trông như "câu trả lời không liên quan"
trong khi thực tế là "metric chưa từng được tính". Sửa bằng cách đoán cũng thất
bại: `gemini-embedding-001` **là** model đúng, nhưng gọi qua
`OpenAIEmbeddings(base_url=...)` vẫn 501. Phải bỏ hẳn tầng đó, gọi SDK native.

**Retry sai nguyên tắc.** Regex chỉ bắt `"retry in X.Xs"`, bỏ qua
`"'retryDelay': '52s'"` → rơi về 10s → chờ 10s rồi lại 429. Tệ hơn, khi quota
ngày đã cạn thì retry vô nghĩa. Log cho phép phân biệt rõ:

| `retryDelay` | Sự cố | Xử lý |
|---|---|---|
| 15–29s | Hết RPM | Chờ rồi thử lại |
| 31–52s | Hết quota ngày | Dừng hẳn, chuyển fallback |

**`parent_id` trùng lặp toàn cục.** `pipeline.py` gọi `chunk_hierarchical()` riêng
cho từng tài liệu, mỗi tài liệu đánh số lại từ 0 → cả 26 tài liệu đều sinh
`parent_0`. Thiết kế "retrieve child → return parent" sẽ trả parent của tài liệu
khác. Test không bắt được vì chỉ chạy trên một đoạn văn bản đơn lẻ.

---

## 3. Quyết định tối ưu lớn nhất

Ban đầu production thua baseline ở `context_recall` (−0.0542). Giả thuyết đầu
tiên là enrichment gây nhiễu — sai.

Ablation có nhóm đối chứng tách được nguyên nhân thật:

```
A: prod chunk (125 đơn vị), hybrid    recall 0.6470
E: baseline chunk (57 đơn vị), hybrid  recall 0.8139   → chênh −0.1668
```

Nguyên nhân nằm ở cách chunk, không phải enrichment: 125 mảnh vụn (median 186
chars, nhỏ nhất 3 chars — nội dung chỉ là `'thu'`) so với 57 đoạn trọn vẹn
(median 383 chars).

Thử 4 phương án đơn vị truy hồi:

| Cấu hình | recall proxy | n | median |
|---|---|---|---|
| child as-is | 0.7310 | 125 | 186 |
| bỏ child <100c | 0.7374 | 100 | 198 |
| **gộp child ≤400c** | **0.8416** | 72 | 297 |

Kết quả thật sau khi sửa: `context_recall` 0.7708 → **0.8667**, vượt baseline
(0.8250).

Điều đáng suy nghĩ nhất: **quyết định hiệu quả nhất không đến từ thêm kỹ thuật
mới, mà từ việc dùng lại đúng cách những thứ đã có sẵn.**

---

## 4. Quyết định thiết kế đáng suy nghĩ

**"Chọn model cho hợp free tier" là câu hỏi sai.** Đọc bảng giá chính thức: mọi
model text đều "Free of charge" ở free tier, nên đổi model không tiết kiệm token
nào. Giới hạn thật là 15 RPM, chung cho mọi model. Câu hỏi đúng là "kiến trúc
gọi API có hợp 15 RPM không" — và câu trả lời là không: 125 request tuần tự là
cách dùng sai.

**Throttle thay vì tăng worker.** Tăng concurrency làm tệ hơn. Throttle biến "bị từ
chối" thành "chậm nhưng chắc chắn xong".

**Khi nào nhận fallback.** Khi lỗi mang tính thời gian (RPD), không phải tốc độ
(RPM). Phân biệt được ngay từ `retryDelay`.

---

## 5. Nếu có thêm 1 tuần

1. **Sửa bước generate — ưu tiên cao nhất.** 3/5 failure có `faithfulness = 0`
   trong khi `context_precision = 1.0` và `context_recall = 1.0`: retrieval hoàn
   hảo, LLM vẫn bịa. Cải thiện truy hồi không giúp được nhóm lỗi này. Cần prompt
   trích nguyên văn, hạ `temperature` 0.3 → 0.1.
2. **Reranker latency.** ~2260ms/20 docs trên CPU, vượt xa ngưỡng 150ms.
3. **Đo độ nhiễu.** Cùng cấu hình cho kết quả khác nhau ~0.05 — đủ che mất chênh
   lệch thật. Chạy ≥3 lần mỗi cấu hình.
4. **Cache RAGAS.** ~15 phút và ~200 lượt Gemini cho mỗi lần đánh giá.
5. **Test cho lỗi xuyên tài liệu** (khắc phục lỗ hổng của `parent_id`).
6. **OCR 2 file PDF scan** đang bị bỏ qua.

---

## 6. Bài học lớn nhất

Phần lớn thời gian không nằm ở viết hàm mà ở việc **làm sao biết mình đã đúng**:
dùng test cho giả định về cấu trúc dữ liệu, dùng ablation có nhóm đối chứng để
tách biến số, và đọc log lỗi thay vì nối dấu vết — ba lỗi Gemini trông không
liên quan nhưng cùng một lớp nguyên nhân.

Hai lần tôi sai đều do cùng một thói quen: tin vào một câu chuyện hợp lý về cơ
chế thay vì kiểm tra xem câu chuyện đó có khớp với dữ liệu không.
