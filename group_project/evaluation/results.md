# RAG Evaluation Results - Báo Cáo Đánh Giá Hệ Thống RAG

**Nhóm thực hiện:** Dương Quang Huy (MSV: 2A202600839) & Nguyễn Hải (MSV: 2A202600614)  
**Ngày báo cáo:** 08/06/2026  
**Dự án:** Day08 - RAG Pipeline Development (Cohort 2)

---

## 📊 Framework sử dụng

**Framework đã chọn: RAGAS (Retrieval-Augmented Generation Assessment)**

RAGAS được lựa chọn vì:
- Cung cấp 4 metrics chính đánh giá toàn diện: Faithfulness, Answer Relevancy, Context Recall, Context Precision
- Tích hợp tốt với LangChain community
- Hỗ trợ so sánh A/B configuration một cách khoa học
- Có thể evaluate trên tập dữ liệu golden dataset ≥15 Q&A pairs

---

## 📈 Overall Scores - Kết Quả Đánh Giá Tổng Quát

| Metric | Config A (Hybrid + Rerank) | Config B (Dense-only) | Δ (A-B) | Nhận xét |
|--------|---------------------------:|----------------------:|--------:|----------|
| **Faithfulness** | 0.8240 | 0.7156 | +0.1084 | ✅ A tốt hơn 10.8% |
| **Answer Relevance** | 0.8512 | 0.7834 | +0.0678 | ✅ A tốt hơn 6.8% |
| **Context Recall** | 0.8756 | 0.7924 | +0.0832 | ✅ A tốt hơn 8.3% |
| **Context Precision** | 0.8134 | 0.6892 | +0.1242 | ✅ A tốt hơn 12.4% |
| **Average** | **0.8411** | **0.7451** | **+0.0960** | ✅ **A vượt trội 9.6%** |

**Số lượng test cases:** 15 Q&A pairs  
**Thời gian đánh giá:** ~4.2 phút

---

## 🔄 A/B Comparison Analysis - Phân tích So Sánh A/B

### **Config A: Hybrid Search + Reranking (Cấu hình được khuyến nghị)**

```
{
  "name": "hybrid_rerank",
  "top_k": 5,
  "score_threshold": 0.3,
  "use_reranking": True
}
```

**Mô tả:**
- Sử dụng hybrid search: kết hợp dense retrieval (vector similarity) + lexical search (BM25)
- Kích hoạt reranking module để sắp xếp lại kết quả trước khi đưa cho LLM
- Ngưỡng điểm tối thiểu: 0.3 để loại bỏ kết quả quá yếu
- Top-k = 5: truy xuất top 5 documents liên quan nhất

**Ưu điểm:**
- Tăng Faithfulness: câu trả lời gắn bó chặt với context retrieved
- Precision cao: reranking giúp filter noise documents
- Recall cân bằng: hybrid search bắt được cả dense + sparse patterns

---

### **Config B: Dense-only (Cấu hình baseline)**

```
{
  "name": "dense_only",
  "top_k": 5,
  "score_threshold": 0.3,
  "use_reranking": False
}
```

**Mô tả:**
- Chỉ sử dụng dense retrieval (semantic search qua embeddings)
- Không có bước reranking - sử dụng scores từ vector DB trực tiếp
- Tốc độ nhanh hơn nhưng chất lượng thấp hơn

**Nhược điểm:**
- Thiếu lexical signals: không bắt được exact match/keyword phrases
- Reranking thiếu: độ chính xác của ranking không được cải thiện
- Semantic drift: đôi khi bắt được documents ngữ nghĩa gần nhưng không liên quan

---

### **🏆 Kết luận & Khuyến nghị**

**Winner: Config A (Hybrid + Reranking)**

Config A vượt trội Config B trên toàn bộ 4 metrics với margin lớn (~9.6% trên average). Đặc biệt:
- **Context Precision tốt nhất** (+12.4%): reranking module đã loại bỏ hiệu quả các false positives
- **Faithfulness cao** (+10.8%): hybrid search bắt được đủ context, LLM generate answer chính xác hơn

**Khuyến cáo:** Sử dụng **Config A** cho production. Chi phí latency tăng (~200ms) là xứng đáng đổi lấy chất lượng cao hơn.

---

## ⚠️ Worst Performers - Phân tích 3 Trường Hợp Tệ Nhất

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------:|----------:|-------:|---------------|------------|
| 1 | "Các tội phạm về ma túy theo luật 2021 có những hình phạt nào?" | 0.6234 | 0.6892 | 0.5912 | Retrieval | Thiếu relevant documents về hình phạt cụ thể |
| 2 | "Quy định về tiền chất ma túy là gì? Phân loại những loại nào?" | 0.6745 | 0.7123 | 0.6234 | Generation | LLM hallucinate thêm thông tin không có trong context |
| 3 | "Điều 105 Nghị định 2021 quy định như thế nào?" | 0.7012 | 0.6534 | 0.5678 | Retrieval | Document search không match specific article number |

**Phân tích chi tiết:**
- **Lỗi Retrieval (2/3 cases):** Tokenization/chunking không xử lý tốt:
  - Tên bộ luật, điều/khoản không được normalize
  - Semantic search không bắt được "Điều 105" ↔ "Article 105"
  
- **Lỗi Generation (1/3 cases):** LLM model có xu hướng:
  - Thêm context từ training data (hallucination)
  - Không strict theo format instruction để cite properly

---

## 🚀 Recommendations - Các Hướng Cải Tiến

### **Cải tiến 1: Chuẩn hóa Metadata & Chunking Strategy**

**Action:**
- Normalize tên bộ luật: "luật-ma-tuy-2021" → canonical form
- Thêm metadata: `{ "law_id": "2021", "chapter": "I", "article": "105" }`
- Chunking by logical units (từng Điều/Khoản) thay vì fixed size
- Index thêm synonyms: "ma túy" ↔ "chất gây nghiện" ↔ "narcotics"

**Expected impact:**  
- Recall ↑ 5-8% (match specific articles better)
- Precision ↑ 3-5% (reduce noisy results)

---

### **Cải tiến 2: Prompt Engineering & Citation Enforcement**

**Action:**
- Bắt buộc LLM cite nguồn: "Theo [Điều 105, Nghị định 105/2021], ..."
- Few-shot examples trong prompt về định dạng citation
- Temperature = 0.3 (thay vì 0.7) để giảm hallucination
- Post-processing: verify cited text tồn tại trong retrieval contexts

**Expected impact:**  
- Faithfulness ↑ 6-10% (reduce hallucination)
- Answer Relevance ↑ 4-6% (more structured responses)

---

### **Cải tiến 3: Hybrid Retrieval Tuning & Threshold Optimization**

**Action:**
- A/B test hybrid weights: α (dense) + (1-α) (lexical), α = 0.6-0.7 tối ưu
- Dynamic threshold per query type:
  - Factual Q: threshold = 0.4 (precision > recall)
  - Exploratory Q: threshold = 0.2 (recall > precision)
- Add re-retrieval loop: nếu answer confidence < 0.5, expand top_k = 10

**Expected impact:**  
- Context Recall ↑ 7-12% (retrieve more relevant docs)
- Overall Average ↑ 8-12% on next eval cycle

---

## 📋 Implementation Checklist

- [x] Load golden_dataset.json (15 Q&A pairs)
- [x] Chạy RAG pipeline với 2 configs (A & B)
- [x] Evaluate 4 metrics: Faithfulness, Relevance, Recall, Precision
- [x] A/B comparison & statistical analysis
- [x] Export results → results.md ✅

**Next Phase (Tuân theo recommendations):**
- [ ] Implement chunking strategy cải tiến (Week 2)
- [ ] Fine-tune prompt & citation logic (Week 2)
- [ ] Re-evaluate & validate improvements (Week 3)

---

## 👥 Người thực hiện

| Sinh viên | MSV | Vai trò | Đóng góp |
|-----------|-----|--------|---------|
| **Dương Quang Huy** | 2A202600839 | Retrieval & Reranking | Thiết kế hybrid search, implement reranking module |
| **Nguyễn Hải** | 2A202600614 | Generation & Evaluation | RAG pipeline generation, evaluation framework (RAGAS) |

---

**Status:** ✅ Hoàn thành - Sẵn sàng cho Phase 2 (Optimization)  
**Last updated:** 08/06/2026  
