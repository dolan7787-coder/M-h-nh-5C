# M-h-nh-5C
# 🏦 Đánh giá Rủi ro Tín dụng Khách hàng vay — Mô hình 5C

Ứng dụng Streamlit chuyển thể từ notebook `Model_cho_vay_kh_nhóm_nợ.ipynb`. Ứng dụng dự
báo khả năng rủi ro (nhóm nợ / PD) của khách hàng vay dựa trên 24 tiêu chí khảo sát theo
mô hình **5C**:

| Nhóm | Ý nghĩa | Các biến |
|---|---|---|
| **TC** | Tư cách (Character) | TC1–TC5 |
| **NL** | Năng lực (Capacity) | NL1–NL4 |
| **DK** | Điều kiện (Condition) | DK1–DK5 |
| **V** | Vốn (Capital) | V1–V6 |
| **TS** | Tài sản đảm bảo (Collateral) | TS1–TS4 |

**Biến mục tiêu:** `PD` — 0 = Không rủi ro, 1 = Có rủi ro.

**Thuật toán:** Logistic Regression (scikit-learn), giống notebook gốc. Không có bước
tiền xử lý (scaler/encoder) nào được áp dụng trong notebook, vì các biến đầu vào đã ở
dạng số nguyên (thang điểm Likert 1–5) nên mô hình dùng trực tiếp dữ liệu thô.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Cấu trúc dữ liệu đầu vào

Tệp CSV cần chứa tối thiểu các cột sau (đúng tên, không phân biệt hoa/thường không được
hỗ trợ — tên cột phải khớp chính xác):

```
TC1, TC2, TC3, TC4, TC5,
NL1, NL2, NL3, NL4,
DK1, DK2, DK3, DK4, DK5,
V1, V2, V3, V4, V5, V6,
TS1, TS2, TS3, TS4,
PD
```

Mỗi biến khảo sát (TC, NL, DK, V, TS) nhận giá trị số nguyên trong thang điểm 1–5.
Cột `PD` là nhãn nhị phân (0/1). File mẫu đính kèm: `5c.csv` (150 dòng khảo sát).

> Ghi chú: cột `NN` có trong file dữ liệu mẫu nhưng **không** được notebook gốc sử dụng
> làm biến đầu vào, nên ứng dụng cũng không dùng cột này.

## Mô tả các tab

1. **⚙️ Sidebar — Cấu hình & Tải dữ liệu:** tải tệp CSV, chọn tỷ lệ tập kiểm tra
   (test size, mặc định 0.2 theo notebook), random state (mặc định 23 theo notebook),
   và các tham số nâng cao của Logistic Regression (C, max_iter, solver — dùng giá trị
   mặc định của scikit-learn vì notebook không chỉnh sửa các tham số này). Bấm
   **"Huấn luyện mô hình"** để chạy.
2. **📋 Tổng quan dữ liệu:** kích thước dữ liệu, xem trước dữ liệu thô, thống kê mô tả
   các biến đưa vào mô hình.
3. **📈 Trực quan hóa dữ liệu:** phân phối biến mục tiêu PD và các biến đầu vào (có thể
   chọn tối đa 3 biến để xem cùng lúc, hiển thị dạng lưới 2×2).
4. **🎯 Kết quả huấn luyện & kiểm định mô hình:** Accuracy, Precision, Recall, F1-score,
   ROC-AUC, ma trận nhầm lẫn, đường cong ROC, và báo cáo phân loại chi tiết. Cần bấm nút
   huấn luyện ở sidebar trước.
5. **🔮 Sử dụng mô hình:** dự báo cho một khách hàng bằng cách nhập trực tiếp điểm khảo
   sát, hoặc dự báo hàng loạt bằng cách tải lên tệp CSV có đúng 24 cột biến đầu vào (kết
   quả có thể tải xuống dưới dạng CSV).

## Ghi chú kỹ thuật

- Yêu cầu Streamlit **≥ 1.38** để đảm bảo tương thích đầy đủ các thành phần giao diện
  đã dùng (`st.container(height=...)`, `st.form`, `st.tabs`).
- Mô hình chỉ huấn luyện lại khi người dùng bấm nút ở sidebar; kết quả được lưu trong
  `st.session_state` để chuyển tab không bị mất dữ liệu hay huấn luyện lại.
