# Hệ thống gợi ý sản phẩm Software trên Amazon (Big Data)

[![Build LaTeX](https://github.com/HThanh-how/customer-churn-prediction-ann/actions/workflows/latex.yml/badge.svg)](https://github.com/HThanh-how/customer-churn-prediction-ann/actions)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5-orange.svg)](https://spark.apache.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)

> **Đồ án môn học: Dữ liệu lớn (Big Data)**
> Trường Đại học Công nghệ Thông tin – ĐHQG TP.HCM
> **GVHD:** Phạm Huy Thanh · **SVTH:** Đào Minh Trí

Hệ thống gợi ý cho nhóm sản phẩm **Software** trên Amazon theo hướng **truy hồi
ngữ nghĩa**: kết hợp **mô hình ngôn ngữ** (sentence embedding), **kiến trúc hai tháp
(Two-Tower)** và **học tương phản (InfoNCE)** trên nền tảng dữ liệu lớn
**Apache Spark**. Dữ liệu: *Amazon Reviews 2023 – Software* (~146K người dùng,
>17K sản phẩm).

---

## 🧠 Kiến trúc

```
                 ┌─────────────────────────┐        ┌──────────────────────────┐
 Lịch sử user →  │ Tháp User: MLP 384→512→  │        │ Tháp Item: MiniLM (đóng   │  ← semantic_text
 (vector 384-d)  │ 384 + BatchNorm + Dropout│        │ băng) + mean pooling      │
                 └────────────┬────────────┘        └────────────┬─────────────┘
                              │  L2 normalize                     │  L2 normalize
                              └──────────► cosine ◄───────────────┘
                                     InfoNCE (in-batch negatives)
```

- **Tháp Item:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, **đóng băng**).
- **Tháp User:** MLP huấn luyện trên vector lịch sử có **trọng số thời gian**
  (recency × rating).
- **Loss:** InfoNCE (τ = 0.1) · **Optimizer:** AdamW · **Mixed precision** FP16.

## 📊 Kết quả (đánh giá full-catalog trên >17K sản phẩm)

| Chỉ số | Two-Tower | Ngẫu nhiên | Bội số |
|---|:---:|:---:|:---:|
| Recall@10 | **0.0142** | 0.00059 | ≈ **24×** |
| Recall@50 | **0.0436** | 0.00294 | ≈ **14×** |
| NDCG@10 | 0.0062 | – | – |

---

## 📁 Cấu trúc dự án

```
report.tex / report.pdf              # Báo cáo LaTeX (XeLaTeX, Times New Roman, 5 chương)
front_matter/  chapters/             # Bìa, lời cảm ơn, tóm tắt, 5 chương + phụ lục
images/                              # Hình minh hoạ (sinh từ scripts/)
scripts/generate_figures.py          # Sinh hình từ số liệu huấn luyện thật
ReSys_xu_ly_du_lieu_pyspark.ipynb    # (1) Tiền xử lý dữ liệu lớn bằng PySpark → Parquet
code_trainning.ipynb                 # (2) Huấn luyện Two-Tower + đánh giá full-catalog
requirements.txt
.github/workflows/latex.yml          # CI tự build PDF + tạo release
```

---

## 🚀 Hướng dẫn sử dụng

```bash
pip install -r requirements.txt
# (1) Tiền xử lý dữ liệu  →  Parquet
jupyter notebook ReSys_xu_ly_du_lieu_pyspark.ipynb
# (2) Huấn luyện & đánh giá mô hình hai tháp
jupyter notebook code_trainning.ipynb
# Sinh lại hình & biên dịch báo cáo
python scripts/generate_figures.py
xelatex report.tex && xelatex report.tex
```
> Hai notebook được tối ưu cho **Google Colab** (GPU). Dữ liệu *Amazon Reviews 2023
> – Software* tải từ <https://amazon-reviews-2023.github.io/>.

---

## 🛠 Công nghệ
Apache Spark (PySpark) · Apache Parquet · PyTorch · Hugging Face Transformers ·
Sentence-Transformers (MiniLM) · InfoNCE contrastive learning · XeLaTeX.

## 🔭 Hướng phát triển
Tinh chỉnh tháp sản phẩm bằng **LoRA** · tích hợp **FAISS** cho gợi ý thời gian thực.
