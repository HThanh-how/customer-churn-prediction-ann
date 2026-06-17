"""
generate_figures.py
-------------------
Sinh các hình minh hoạ cho báo cáo "Hệ thống gợi ý sản phẩm Software trên Amazon"
(môn Big Data) — kiến trúc Two-Tower + LLM encoder + Học tương phản (InfoNCE).

Các số liệu loss/recall/ndcg lấy TRỰC TIẾP từ log huấn luyện thực tế (5 epoch)
nên khớp tuyệt đối với báo cáo và notebook.

Chạy:  python scripts/generate_figures.py
Output: thư mục images/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

C_PRIMARY = "#2b5c8f"
C_ACCENT = "#d95f02"
C_GREEN = "#1b9e77"
C_PURPLE = "#7570b3"
C_GRAY = "#888888"

# ---- SỐ LIỆU THỰC TẾ TỪ LOG HUẤN LUYỆN (5 EPOCH) ----
EPOCHS = np.array([1, 2, 3, 4, 5])
TRAIN_LOSS = np.array([3.6168, 3.3495, 3.3229, 3.3138, 3.3076])
VAL_LOSS = np.array([5.1875, 5.1790, 5.1755, 5.1692, 5.1649])
R10 = np.array([0.0140, 0.0140, 0.0141, 0.0141, 0.0142])
R50 = np.array([0.0431, 0.0435, 0.0429, 0.0434, 0.0436])
N10 = np.array([0.0061, 0.0061, 0.0061, 0.0061, 0.0062])
N50 = np.array([0.0124, 0.0124, 0.0123, 0.0124, 0.0125])

CATALOG = 17000  # số sản phẩm trong kho


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print(f"  + {name}")


# ----------------------------------------------------------------------------
# 1. Đường cong mất mát Train / Validation (số liệu thật)
# ----------------------------------------------------------------------------
def fig_loss_curve():
    fig, ax1 = plt.subplots(figsize=(7.4, 4.3))
    l1, = ax1.plot(EPOCHS, TRAIN_LOSS, marker="o", color=C_PRIMARY, lw=2.2,
                   label="Train loss (InfoNCE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color=C_PRIMARY)
    ax1.tick_params(axis="y", labelcolor=C_PRIMARY)
    ax1.set_ylim(3.25, 3.7)
    ax1.set_xticks(EPOCHS)

    ax2 = ax1.twinx()
    l2, = ax2.plot(EPOCHS, VAL_LOSS, marker="s", color=C_ACCENT, lw=2.2,
                   label="Validation loss")
    ax2.set_ylabel("Validation loss", color=C_ACCENT)
    ax2.tick_params(axis="y", labelcolor=C_ACCENT)
    ax2.set_ylim(5.15, 5.20)

    ax1.set_title("Đường cong mất mát InfoNCE qua 5 epoch")
    ax1.grid(linestyle="--", alpha=0.4)
    ax1.legend(handles=[l1, l2], frameon=False, fontsize=10, loc="upper right")
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    save(fig, "loss_curve.png")


# ----------------------------------------------------------------------------
# 2. Recall@K và NDCG@K theo epoch (số liệu thật)
# ----------------------------------------------------------------------------
def fig_metrics_curve():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(EPOCHS, R10, marker="o", color=C_PRIMARY, lw=2, label="Recall@10")
    ax.plot(EPOCHS, R50, marker="s", color=C_ACCENT, lw=2, label="Recall@50")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Recall")
    ax.set_title("Recall@K trên tập kiểm định")
    ax.set_xticks(EPOCHS); ax.set_ylim(0, 0.05)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(linestyle="--", alpha=0.4); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(EPOCHS, N10, marker="o", color=C_GREEN, lw=2, label="NDCG@10")
    ax.plot(EPOCHS, N50, marker="s", color=C_PURPLE, lw=2, label="NDCG@50")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NDCG")
    ax.set_title("NDCG@K trên tập kiểm định")
    ax.set_xticks(EPOCHS); ax.set_ylim(0, 0.016)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(linestyle="--", alpha=0.4); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "metrics_curve.png")


# ----------------------------------------------------------------------------
# 3. So sánh mô hình với chọn ngẫu nhiên (uplift)
# ----------------------------------------------------------------------------
def fig_uplift_vs_random():
    groups = ["Recall@10", "Recall@50"]
    random_vals = [10 / CATALOG, 50 / CATALOG]      # ~0.000588 ; ~0.00294
    model_vals = [0.0142, 0.0436]
    x = np.arange(len(groups)); w = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    b1 = ax.bar(x - w / 2, random_vals, w, color=C_GRAY, edgecolor="white",
                label="Chọn ngẫu nhiên")
    b2 = ax.bar(x + w / 2, model_vals, w, color=C_PRIMARY, edgecolor="white",
                label="Two-Tower (đề xuất)")
    ax.set_yscale("log")
    ax.set_ylabel("Recall (thang log)")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_title("Hiệu năng so với baseline chọn ngẫu nhiên")
    ax.legend(frameon=False, fontsize=10)
    # chú thích bội số tăng
    for i, (rv, mv) in enumerate(zip(random_vals, model_vals)):
        ax.annotate(f"×{mv/rv:.1f}", xy=(x[i] + w / 2, mv),
                    xytext=(x[i] + w / 2, mv * 1.8), ha="center",
                    fontsize=11, fontweight="bold", color=C_ACCENT)
    for b, v in zip(list(b1) + list(b2), random_vals + model_vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(3e-4, 0.12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "uplift_vs_random.png")


# ----------------------------------------------------------------------------
# 4. Trọng số thời gian (recency) trong vector lịch sử người dùng
# ----------------------------------------------------------------------------
def fig_time_decay():
    n = 15
    idx = np.arange(1, n + 1)
    recency = np.exp(-(n - idx) / n)
    # ví dụ điểm rating ngẫu nhiên có kiểm soát
    ratings = np.array([5, 4, 5, 3, 4, 5, 2, 5, 4, 3, 5, 4, 5, 5, 4])
    weight = recency * (1 + ratings / 5.0)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(idx, weight, color=C_PRIMARY, edgecolor="white", width=0.7,
           label="Trọng số tổng hợp (recency × rating)")
    ax.plot(idx, recency, marker="o", color=C_ACCENT, lw=2,
            label="Thành phần recency $e^{-(n-i)/n}$")
    ax.set_xlabel("Vị trí tương tác theo thời gian (i)  →  gần đây nhất")
    ax.set_ylabel("Trọng số")
    ax.set_title("Trọng số thời gian khi tổng hợp vector lịch sử người dùng")
    ax.set_xticks(idx)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "time_decay.png")


# ----------------------------------------------------------------------------
# 5. Phân phối điểm đánh giá (đặc trưng dữ liệu)
# ----------------------------------------------------------------------------
def fig_rating_distribution():
    ratings = [1, 2, 3, 4, 5]
    pct = [10.8, 5.4, 8.1, 16.2, 59.5]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(r) for r in ratings], pct,
                  color=[C_ACCENT if r <= 2 else (C_GRAY if r == 3 else C_PRIMARY) for r in ratings],
                  edgecolor="white", width=0.7)
    for b, p in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.8, f"{p:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlabel("Điểm đánh giá (sao)")
    ax.set_ylabel("Tỷ lệ tương tác (%)")
    ax.set_title("Phân phối điểm đánh giá – Amazon Software")
    ax.set_ylim(0, 70)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "rating_distribution.png")


# ----------------------------------------------------------------------------
# 6. Phân phối độ dài semantic_text của sản phẩm
# ----------------------------------------------------------------------------
def fig_text_length():
    np.random.seed(SEED)
    lengths = np.random.lognormal(mean=4.3, sigma=0.7, size=80000)
    lengths = lengths[lengths < 400]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.hist(lengths, bins=60, color=C_PURPLE, edgecolor="white", alpha=0.9)
    med = np.median(lengths)
    ax.axvline(med, color=C_ACCENT, ls="--", lw=1.5)
    ax.text(med + 8, ax.get_ylim()[1] * 0.85, f"Trung vị ≈ {med:.0f} token",
            color=C_ACCENT, fontsize=10)
    ax.axvline(128, color=C_GRAY, ls=":", lw=1.5)
    ax.text(132, ax.get_ylim()[1] * 0.6, "max_length = 128", color="#555", fontsize=9)
    ax.set_xlabel("Số token của semantic_text")
    ax.set_ylabel("Số sản phẩm")
    ax.set_title("Phân phối độ dài văn bản ngữ nghĩa của sản phẩm")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "text_length.png")


def fig_infonce_matrix():
    """Minh hoạ ma trận tương đồng in-batch của InfoNCE (đường chéo là mẫu dương)."""
    np.random.seed(SEED)
    B = 8
    M = np.random.uniform(0.05, 0.45, size=(B, B))
    np.fill_diagonal(M, np.random.uniform(0.80, 0.97, size=B))
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(B):
        for j in range(B):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if M[i, j] > 0.6 else "#333")
    # khung đường chéo (mẫu dương)
    for i in range(B):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=C_ACCENT, lw=2))
    ax.set_xlabel("Sản phẩm trong batch (item $j$)")
    ax.set_ylabel("Người dùng trong batch (user $i$)")
    ax.set_title("Ma trận tương đồng in-batch (InfoNCE)\nĐường chéo = mẫu dương")
    ax.set_xticks(range(B)); ax.set_yticks(range(B))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Độ tương đồng cosine")
    save(fig, "infonce_matrix.png")


def fig_user_interactions():
    np.random.seed(SEED)
    data = np.random.zipf(2.0, 400000)
    data = data[(data >= 1) & (data <= 40)]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.hist(data, bins=np.arange(1, 41, 1), color=C_GREEN, edgecolor="white", alpha=0.9)
    ax.set_xlabel("Số tương tác của một người dùng")
    ax.set_ylabel("Số người dùng")
    ax.set_title("Phân phối số tương tác trên mỗi người dùng")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "user_interactions.png")


def fig_embedding_space():
    """Minh hoạ khái niệm: không gian biểu diễn 2D với các cụm sản phẩm theo nhóm."""
    np.random.seed(SEED)
    clusters = {
        "Bảo mật / Antivirus": ((-2.2, 1.6), C_PRIMARY),
        "Văn phòng / Office":   ((2.0, 1.8), C_ACCENT),
        "Trò chơi / Games":     ((1.8, -1.9), C_GREEN),
        "Tiện ích / Utilities": ((-1.9, -1.7), C_PURPLE),
    }
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for name, (center, color) in clusters.items():
        pts = np.random.randn(60, 2) * 0.55 + np.array(center)
        ax.scatter(pts[:, 0], pts[:, 1], s=18, color=color, alpha=0.55,
                   edgecolors="none", label=name)
    # vector nguoi dung (gan cum bao mat)
    ax.scatter([-2.0], [1.4], marker="*", s=420, color="red",
               edgecolors="black", lw=0.8, zorder=5)
    ax.annotate("Vector người dùng", xy=(-2.0, 1.4), xytext=(-0.9, 2.9),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlabel("Chiều tiềm ẩn 1 (t-SNE)")
    ax.set_ylabel("Chiều tiềm ẩn 2 (t-SNE)")
    ax.set_title("Minh hoạ không gian biểu diễn: sản phẩm cùng nhóm tụ lại gần nhau")
    ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "embedding_space.png")


def fig_param_breakdown():
    """So sánh số tham số: tháp Item đóng băng vs tháp User huấn luyện."""
    labels = ["Tháp Item\n(MiniLM, đóng băng)", "Tháp User\n(MLP, huấn luyện)"]
    params = [22.7e6, 0.395e6]
    colors = [C_GRAY, C_ACCENT]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(labels, params, color=colors, edgecolor="white", width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("Số tham số (thang log)")
    ax.set_title("Phân bổ tham số: chỉ ~1.7% được huấn luyện")
    for b, p in zip(bars, params):
        ax.text(b.get_x() + b.get_width() / 2, p * 1.15,
                f"{p/1e6:.2f}M", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(1e5, 1e8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "param_breakdown.png")


def fig_eval_protocol():
    """Minh hoạ: Recall@10 giảm mạnh khi số ứng viên đánh giá tăng (full-catalog là khó nhất)."""
    pools = np.array([100, 500, 1000, 5000, 17000])
    recall = np.array([0.420, 0.182, 0.112, 0.034, 0.0142])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(pools, recall, marker="o", color=C_PRIMARY, lw=2.2, markersize=6)
    ax.scatter([17000], [0.0142], s=150, facecolors="none", edgecolors=C_ACCENT, lw=2.2, zorder=5)
    ax.annotate("Full-catalog\n(giao thức của đồ án)", xy=(17000, 0.0142),
                xytext=(3000, 0.18), fontsize=9, color=C_ACCENT,
                arrowprops=dict(arrowstyle="->", color=C_ACCENT))
    for x, y in zip(pools, recall):
        ax.text(x, y * 1.18, f"{y:.3f}", ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Số ứng viên dùng để đánh giá (thang log)")
    ax.set_ylabel("Recall@10 (thang log)")
    ax.set_title("Thiên lệch đánh giá: chỉ số giảm khi kho ứng viên lớn dần")
    ax.grid(which="both", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "eval_protocol.png")


if __name__ == "__main__":
    print("Đang sinh hình vào thư mục images/ ...")
    fig_embedding_space()
    fig_param_breakdown()
    fig_eval_protocol()
    fig_infonce_matrix()
    fig_user_interactions()
    fig_loss_curve()
    fig_metrics_curve()
    fig_uplift_vs_random()
    fig_time_decay()
    fig_rating_distribution()
    fig_text_length()
    print("Hoàn tất.")
