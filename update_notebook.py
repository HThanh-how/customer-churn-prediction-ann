import nbformat as nbf

# Tạo một notebook mới
nb = nbf.v4.new_notebook()

# Danh sách các cell
cells = []

# Cell 1: Markdown giới thiệu bài làm
cells.append(nbf.v4.new_markdown_cell("""# Lab 4.1: Xây dựng ANN với Tensorflow dự đoán Churn Customer
**Người thực hiện:** Đào Minh Trí
**Đề tài:** Dự đoán khả năng rời bỏ dịch vụ của khách hàng (Customer Churn) bằng mạng nơ-ron nhân tạo (Artificial Neural Network) sử dụng thư viện TensorFlow và Keras.

---
### Các bước thực hiện theo yêu cầu:
1. Kết nối Google Drive và cài đặt/gọi thư viện.
2. Tải dữ liệu và cấu hình đường dẫn đọc dữ liệu từ Google Drive/Local.
3. Xây dựng mô hình mạng nơ-ron nhân tạo (ANN).
4. Huấn luyện (đào tạo) mô hình.
5. Kiểm tra và đánh giá hiệu năng mô hình.
6. Sử dụng mô hình dự đoán.
7. Hiển thị kết quả trực quan (đồ thị Loss/Accuracy và Confusion Matrix).
"""))

# Cell 2: Markdown mục 1
cells.append(nbf.v4.new_markdown_cell("""## 1. Kết nối Google Drive, Cài đặt và Gọi thư viện"""))

# Cell 3: Code kết nối Google Drive
cells.append(nbf.v4.new_code_cell("""# 1. Kết nối Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Kết nối Google Drive thành công!")
except ImportError:
    print("Không phát hiện môi trường Google Colab. Bỏ qua kết nối Google Drive.")
"""))

# Cell 4: Code cài đặt thư viện
cells.append(nbf.v4.new_code_cell("""# 2. Cài đặt các thư viện cần thiết
!pip install tensorflow pandas numpy scikit-learn matplotlib seaborn kagglehub
"""))

# Cell 5: Code import thư viện
cells.append(nbf.v4.new_code_cell("""# 3. Gọi các thư viện
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print("TensorFlow Version:", tf.__version__)
"""))

# Cell 6: Markdown mục 2
cells.append(nbf.v4.new_markdown_cell("""## 2. Tải dữ liệu và Thiết lập đường dẫn đọc dữ liệu"""))

# Cell 7: Code tải dữ liệu qua kagglehub
cells.append(nbf.v4.new_code_cell("""# 4. Tải dữ liệu từ Kaggle
import kagglehub
try:
    path = kagglehub.dataset_download("aakash50897/churn-modellingcsv")
    print("Đường dẫn tải dữ liệu về máy:", path)
except Exception as e:
    print("Lỗi khi tải dữ liệu:", e)
"""))

# Cell 8: Code thiết lập đường dẫn và đọc dữ liệu từ Drive/Local
cells.append(nbf.v4.new_code_cell("""# 5. Thiết lập đường dẫn và đọc dữ liệu từ Google Drive (hoặc file local)
drive_path = '/content/drive/MyDrive/Churn_Modelling.csv'
local_path = 'Churn_Modelling.csv'

# Kiểm tra đường dẫn nào tồn tại để đọc dữ liệu
if os.path.exists(drive_path):
    csv_path = drive_path
    print(f"Đọc dữ liệu từ Google Drive: {csv_path}")
else:
    # Nếu chạy local hoặc chưa upload lên Drive, copy file đã tải về thư mục hiện tại để đọc
    if not os.path.exists(local_path):
        import shutil
        source_file = os.path.join(path, 'Churn_Modelling.csv')
        if os.path.exists(source_file):
            shutil.copy(source_file, local_path)
    csv_path = local_path
    print(f"Đọc dữ liệu từ file local: {csv_path}")

# Đọc dữ liệu vào DataFrame
dataset = pd.read_csv(csv_path)
dataset.head()
"""))

# Cell 9: Code thông tin cơ bản của dữ liệu
cells.append(nbf.v4.new_code_cell("""# Hiển thị thông tin tổng quan về tập dữ liệu
dataset.info()
"""))

# Cell 10: Markdown tách cột
cells.append(nbf.v4.new_markdown_cell("""Tách các cột không cần thiết như `RowNumber`, `CustomerId`, `Surname` và lấy biến phụ thuộc `Exited` (biến đích biểu thị khách hàng rời đi hay ở lại).
"""))

# Cell 11: Code tách cột độc lập và phụ thuộc
cells.append(nbf.v4.new_code_cell("""X = dataset.drop(labels=['CustomerId', 'Surname', 'RowNumber', 'Exited'], axis=1)
y = dataset['Exited']
X.head()
"""))

# Cell 12: Markdown mã hóa
cells.append(nbf.v4.new_markdown_cell("""Mã hóa các cột dạng phân loại (categorical variables) gồm `Geography` (Quốc gia) và `Gender` (Giới tính) sử dụng `LabelEncoder`.
"""))

# Cell 13: Code mã hóa biến phân loại
cells.append(nbf.v4.new_code_cell("""# Khởi tạo và mã hóa cột Geography
label_geo = LabelEncoder()
X['Geography'] = label_geo.fit_transform(X['Geography'])

# Khởi tạo và mã hóa cột Gender
label_gender = LabelEncoder()
X['Gender'] = label_gender.fit_transform(X['Gender'])

X.head()
"""))

# Cell 14: Markdown One-Hot Encoding
cells.append(nbf.v4.new_markdown_cell("""Do quốc gia (`Geography`) có 3 nhóm giá trị, sau khi Label Encoding, ta thực hiện One-Hot Encoding sử dụng `pd.get_dummies` với tham số `drop_first=True` để tránh bẫy đa cộng tuyến (Dummy Variable Trap).
"""))

# Cell 15: Code One-Hot Encoding
cells.append(nbf.v4.new_code_cell("""X = pd.get_dummies(X, drop_first=True, columns=['Geography'])
X.head()
"""))

# Cell 16: Markdown chia dữ liệu train-test
cells.append(nbf.v4.new_markdown_cell("""Chia tập dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%), phân tầng (stratify) dựa trên biến đích `y` để đảm bảo tỷ lệ nhãn cân đối.
"""))

# Cell 17: Code chia dữ liệu
cells.append(nbf.v4.new_code_cell("""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
print("Kích thước tập Train:", X_train.shape)
print("Kích thước tập Test:", X_test.shape)
"""))

# Cell 18: Markdown chuẩn hóa dữ liệu
cells.append(nbf.v4.new_markdown_cell("""Chuẩn hóa các đặc trưng (Feature Scaling) bằng `StandardScaler` để đưa các giá trị về cùng một phân phối chuẩn có trung bình là 0 và độ lệch chuẩn là 1, giúp mạng ANN hội tụ nhanh hơn.
"""))

# Cell 19: Code chuẩn hóa đặc trưng
cells.append(nbf.v4.new_code_cell("""scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train[:3]
"""))

# Cell 20: Markdown mục 3: Xây dựng mô hình
cells.append(nbf.v4.new_markdown_cell("""## 3. Xây dựng mô hình mạng ANN (Build ANN)
Khởi tạo mô hình mạng nơ-ron nhân tạo tuần tự (`Sequential`) bao gồm:
- **Lớp đầu tiên (Input & Hidden Layer 1)**: Số nút bằng kích thước đầu vào (`X.shape[1]`), hàm kích hoạt (activation function) là `relu`.
- **Lớp ẩn thứ hai (Hidden Layer 2)**: 128 nút, hàm kích hoạt `relu`.
- **Lớp đầu ra (Output Layer)**: 1 nút (do dự đoán nhị phân), hàm kích hoạt `sigmoid` để đưa ra xác suất rời bỏ dịch vụ.
"""))

# Cell 21: Code xây dựng mô hình
cells.append(nbf.v4.new_code_cell("""model = Sequential()
model.add(Dense(X.shape[1], activation='relu', input_dim=X.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
"""))

# Cell 22: Markdown biên dịch mô hình
cells.append(nbf.v4.new_markdown_cell("""Biên dịch (Compile) mô hình:
- Thuật toán tối ưu (Optimizer): `adam`
- Hàm mất mát (Loss function): `binary_crossentropy`
- Chỉ số đánh giá (Metrics): `accuracy`
"""))

# Cell 23: Code biên dịch mô hình
cells.append(nbf.v4.new_code_cell("""model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
"""))

# Cell 24: Markdown mục 4: Đào tạo mô hình
cells.append(nbf.v4.new_markdown_cell("""## 4. Đào tạo mô hình (Train/Fit ANN)
Huấn luyện mô hình trên tập huấn luyện (`X_train`) trong 10 epoch, batch size bằng 10, phân tách thêm 10% tập validation để theo dõi quá trình học tập.
"""))

# Cell 25: Code huấn luyện mô hình
cells.append(nbf.v4.new_code_cell("""history = model.fit(
    X_train, 
    y_train.to_numpy(), 
    batch_size=10, 
    epochs=10, 
    validation_split=0.1, 
    verbose=1
)
"""))

# Cell 26: Markdown mục 5 & 6: Sử dụng và Kiểm tra mô hình
cells.append(nbf.v4.new_markdown_cell("""## 5. Kiểm tra và Sử dụng mô hình để dự đoán (Evaluation & Prediction)
Sử dụng mô hình để dự đoán nhãn lớp cho tập kiểm tra (`X_test`).
"""))

# Cell 27: Code dự đoán (Sử dụng mô hình)
cells.append(nbf.v4.new_code_cell("""y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype("int32")
"""))

# Cell 28: Code đánh giá mô hình bằng phương thức evaluate (Kiểm tra mô hình)
cells.append(nbf.v4.new_code_cell("""test_loss, test_acc = model.evaluate(X_test, y_test.to_numpy())
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
"""))

# Cell 29: Code hiển thị các chỉ số đánh giá chi tiết
cells.append(nbf.v4.new_code_cell("""# Tạo ma trận nhầm lẫn và hiển thị các chỉ số đánh giá
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\\n", cm)
print("Accuracy Score:", acc)
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))
"""))

# Cell 30: Markdown mục 7: Trực quan hóa kết quả
cells.append(nbf.v4.new_markdown_cell("""## 6. Hiển thị kết quả trực quan (Visualization)
Vẽ đồ thị lịch sử huấn luyện (Loss và Accuracy qua từng epoch) và Heatmap của Ma trận nhầm lẫn.
"""))

# Cell 31: Code vẽ lịch sử huấn luyện
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(12, 5))

# Đồ thị Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='#2b5c8f', lw=2)
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Val Loss', color='#d95f02', lw=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Đồ thị Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='#2b5c8f', lw=2)
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#d95f02', lw=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
plt.show()
"""))

# Cell 32: Code vẽ ma trận nhầm lẫn
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Stay', 'Churn'],
            yticklabels=['Stay', 'Churn'],
            annot_kws={'size': 14, 'weight': 'bold'})
plt.title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Actual Class', fontsize=12)
		
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
"""))

# Gán danh sách cells vào notebook
nb['cells'] = cells

# Ghi notebook ra file
with open('churn_modelling.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Đã cập nhật cấu hình Colab & Drive vào churn_modelling.ipynb thành công!")
