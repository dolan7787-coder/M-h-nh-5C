"""
Ứng dụng Chấm điểm Rủi ro Tín dụng Khách hàng theo Mô hình 5C
Chuyển đổi từ notebook 5C_model.ipynb (Logistic Regression)
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

# ============================================================
# 1) CẤU HÌNH TRANG (phải là lệnh Streamlit đầu tiên)
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="Chấm điểm rủi ro tín dụng - Mô hình 5C",
    page_icon="🏦",
)

# ============================================================
# 2) HẰNG SỐ & HÀM DÙNG CHUNG
# ============================================================

# Nhóm biến theo khung 5C (đọc từ notebook: X = 24 cột thang Likert 1-5)
FEATURE_GROUPS = {
    "TC – Tính cách (Character)": ["TC1", "TC2", "TC3", "TC4", "TC5"],
    "NL – Năng lực (Capacity)": ["NL1", "NL2", "NL3", "NL4"],
    "DK – Điều kiện (Condition)": ["DK1", "DK2", "DK3", "DK4", "DK5"],
    "V – Vốn (Capital)": ["V1", "V2", "V3", "V4", "V5", "V6"],
    "TS – Tài sản đảm bảo (Collateral)": ["TS1", "TS2", "TS3", "TS4"],
}
FEATURE_COLS = [c for group in FEATURE_GROUPS.values() for c in group]  # đúng thứ tự notebook
TARGET_COL = "PD"
TARGET_LABELS = {0: "Không có rủi ro", 1: "Có rủi ro"}


def _try_standard_read(raw_bytes: bytes):
    """Thử đọc bằng pandas.read_csv với các encoding phổ biến."""
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
            if TARGET_COL in df.columns and set(FEATURE_COLS).issubset(df.columns):
                return df
        except Exception:
            continue
    return None


def _try_quoted_semicolon_parse(raw_bytes: bytes):
    """
    Dự phòng cho định dạng dữ liệu mẫu đặc thù: mỗi dòng thực chất là một
    chuỗi CSV (phân tách bằng dấu phẩy) được bọc trong dấu ngoặc kép, theo
    sau bởi các dấu ';' và ghi chú phụ. Cần trích đúng phần trong ngoặc kép.
    """
    for enc in ("cp1258", "utf-8-sig", "utf-8", "latin1"):
        try:
            text = raw_bytes.decode(enc)
        except Exception:
            continue
        lines = [ln for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
        rows = []
        ok = True
        for ln in lines:
            if not ln.startswith('"'):
                ok = False
                break
            end = ln.find('"', 1)
            if end == -1:
                ok = False
                break
            rows.append(ln[1:end].split(","))
        if not ok or len(rows) < 2:
            continue
        header, data = rows[0], rows[1:]
        if len(set(len(r) for r in data)) != 1:
            continue
        df = pd.DataFrame(data, columns=header)
        if TARGET_COL in df.columns and set(FEATURE_COLS).issubset(df.columns):
            # cột đầu là mốc thời gian dạng text, các cột còn lại là số
            for c in df.columns[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
    return None


@st.cache_data(show_spinner=False)
def load_data(raw_bytes: bytes) -> pd.DataFrame:
    """Nạp dữ liệu từ bytes file upload, trả về DataFrame sạch (đã ép kiểu số)."""
    df = _try_standard_read(raw_bytes)
    if df is None:
        df = _try_quoted_semicolon_parse(raw_bytes)
    if df is None:
        raise ValueError(
            "Không đọc được file. Vui lòng kiểm tra định dạng CSV và đảm bảo "
            "có đủ các cột biến đầu vào (TC1..TS4) và cột mục tiêu 'PD'."
        )
    # Chuẩn hoá kiểu dữ liệu cho các cột mô hình
    for c in FEATURE_COLS + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def validate_columns(df: pd.DataFrame):
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    return missing


# ============================================================
# 3) SIDEBAR — CẤU HÌNH
# ============================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")

    uploaded_file = st.file_uploader(
        "Tải file dữ liệu (.csv)",
        type=["csv"],
        help="File CSV chứa các cột biến 5C (TC1..TS4) và cột mục tiêu PD, giống cấu trúc file 5c.csv mẫu.",
    )

    # Chỉ có 1 thuật toán duy nhất trong notebook (Logistic Regression)
    # → không hiển thị lựa chọn mô hình.

    st.subheader("Tham số mô hình AI")

    test_size = st.slider(
        "Tỷ lệ tập kiểm tra (test_size)",
        min_value=0.1, max_value=0.5, value=0.2, step=0.05,
        help="Tỷ lệ dữ liệu dùng để kiểm định mô hình, mặc định theo notebook là 0.2.",
    )
    random_state_split = st.number_input(
        "random_state (chia tập train/test)",
        min_value=0, value=23, step=1,
        help="Giá trị hạt giống ngẫu nhiên khi chia tập dữ liệu, mặc định theo notebook là 23.",
    )

    with st.expander("Tham số nâng cao của Logistic Regression"):
        C = st.slider(
            "C (nghịch đảo cường độ regularization)",
            min_value=0.01, max_value=10.0, value=1.0, step=0.01,
            help="Giá trị càng nhỏ, regularization càng mạnh. Mặc định scikit-learn là 1.0.",
        )
        max_iter = st.number_input(
            "max_iter (số vòng lặp tối đa)",
            min_value=100, max_value=5000, value=100, step=100,
            help="Số vòng lặp tối đa để thuật toán hội tụ. Mặc định scikit-learn là 100.",
        )
        solver = st.selectbox(
            "solver",
            options=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            index=0,
            help="Thuật toán tối ưu hoá dùng để huấn luyện Logistic Regression.",
        )
        set_model_seed = st.checkbox(
            "Đặt random_state cho mô hình", value=False,
            help="Notebook gốc không đặt random_state cho model (mặc định None).",
        )
        random_state_model = None
        if set_model_seed:
            random_state_model = st.number_input(
                "random_state (mô hình)", min_value=0, value=23, step=1
            )

    st.divider()
    train_clicked = st.button(
        "🚀 Huấn luyện mô hình", type="primary", use_container_width=True
    )

# ============================================================
# 4) HEADER — VÙNG ĐỊNH HƯỚNG
# ============================================================
st.title("🏦 Chấm điểm Rủi ro Tín dụng Khách hàng theo Mô hình 5C")
st.caption(
    "Ứng dụng huấn luyện mô hình Logistic Regression để dự đoán rủi ro tín dụng "
    "khách hàng (PD: 0 = không có rủi ro, 1 = có rủi ro) dựa trên 24 tiêu chí "
    "đánh giá theo khung 5C: Tính cách, Năng lực, Điều kiện, Vốn, Tài sản đảm bảo "
    "(thang điểm Likert 1-5). Tải lên file CSV dữ liệu khách hàng để bắt đầu."
)

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên file dữ liệu CSV ở thanh bên trái để bắt đầu.")
    st.stop()

file_bytes = uploaded_file.getvalue()

try:
    df = load_data(file_bytes)
except Exception as e:
    st.error(f"❌ Lỗi khi đọc dữ liệu: {e}")
    st.stop()

missing_cols = validate_columns(df)
if missing_cols:
    st.error(f"❌ Dữ liệu thiếu các cột bắt buộc: {', '.join(missing_cols)}")
    st.stop()

if df.empty:
    st.error("❌ Dữ liệu rỗng sau khi làm sạch. Vui lòng kiểm tra lại file.")
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}** — {df.shape[0]} dòng, {len(FEATURE_COLS)} biến đầu vào.")
st.divider()

# ============================================================
# 5) HUẤN LUYỆN MÔ HÌNH (chỉ chạy khi bấm nút, lưu session_state)
# ============================================================
if train_clicked:
    with st.spinner("Đang huấn luyện mô hình..."):
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state_split)
        )

        model = LogisticRegression(
            C=C, max_iter=int(max_iter), solver=solver,
            random_state=random_state_model,
        )
        model.fit(X_train, y_train)

        yhat_test = model.predict(X_test)
        yproba_test = model.predict_proba(X_test)[:, 1]

        st.session_state["model"] = model
        st.session_state["preprocessor"] = None  # notebook không dùng scaler/encoder
        st.session_state["results"] = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "yhat_test": yhat_test, "yproba_test": yproba_test,
            "feature_cols": FEATURE_COLS,
        }
    st.success("✅ Huấn luyện mô hình thành công! Xem kết quả ở tab 'Kết quả huấn luyện & kiểm định mô hình'.")

# ============================================================
# 6) CÁC TAB NỘI DUNG CHÍNH
# ============================================================
tab_overview, tab_viz, tab_result, tab_use = st.tabs(
    ["📋 Tổng quan dữ liệu", "📊 Trực quan hóa dữ liệu", "🎯 Kết quả huấn luyện & kiểm định mô hình", "🔮 Sử dụng mô hình"]
)

# ------------------------------------------------------------
# TAB 3: TỔNG QUAN DỮ LIỆU
# ------------------------------------------------------------
with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng", df.shape[0])
    c2.metric("Số cột", df.shape[1])
    c3.metric("Dung lượng file", f"{len(file_bytes) / (1024*1024):.2f} MB")

    st.subheader("Xem dữ liệu thô")
    with st.container(height=300):
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Thống kê mô tả các biến của mô hình (X và y)")
    st.dataframe(df[FEATURE_COLS + [TARGET_COL]].describe().T, use_container_width=True)

# ------------------------------------------------------------
# TAB 4: TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------------------------
with tab_viz:
    st.caption(
        "Biến mục tiêu PD được hiển thị trước, sau đó là các biến đầu vào theo khung 5C "
        "(thang Likert 1-5 → biểu đồ tần suất)."
    )

    default_vars = ["TC1", "NL1", "V1"]
    selected_vars = st.multiselect(
        "Chọn thêm biến đầu vào để trực quan hóa (tối đa khuyến nghị 3 biến)",
        options=FEATURE_COLS,
        default=default_vars,
        help="Có 24 biến đầu vào; chọn tối đa 3 biến để hiển thị cùng biến mục tiêu trong lưới 2x2.",
    )
    plot_vars = selected_vars[:3] if selected_vars else default_vars[:3]

    grid_items = [("target", TARGET_COL)] + [("feature", v) for v in plot_vars]

    rows_of_two = [grid_items[i:i + 2] for i in range(0, len(grid_items), 2)]
    for row in rows_of_two:
        cols = st.columns(2)
        for col, (kind, varname) in zip(cols, row):
            with col:
                if kind == "target":
                    vc = df[TARGET_COL].map(TARGET_LABELS).value_counts().reset_index()
                    vc.columns = ["Nhãn", "Số lượng"]
                    fig = px.bar(
                        vc, x="Nhãn", y="Số lượng",
                        title="Phân phối biến mục tiêu (PD)",
                        color="Nhãn",
                        color_discrete_map={"Không có rủi ro": "#2ca02c", "Có rủi ro": "#d62728"},
                    )
                else:
                    vc = df[varname].value_counts().sort_index().reset_index()
                    vc.columns = ["Điểm", "Số lượng"]
                    vc["Điểm"] = vc["Điểm"].astype(str)
                    fig = px.bar(vc, x="Điểm", y="Số lượng", title=f"Phân phối điểm {varname}")
                fig.update_layout(height=350, margin=dict(t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TAB 5: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# ------------------------------------------------------------
with tab_result:
    if "results" not in st.session_state:
        st.info("ℹ️ Chưa có mô hình nào được huấn luyện. Vui lòng bấm nút '🚀 Huấn luyện mô hình' ở thanh bên trái.")
    else:
        res = st.session_state["results"]
        y_test = res["y_test"]
        yhat_test = res["yhat_test"]
        yproba_test = res["yproba_test"]

        acc = accuracy_score(y_test, yhat_test)
        prec = precision_score(y_test, yhat_test, zero_division=0)
        rec = recall_score(y_test, yhat_test, zero_division=0)
        f1 = f1_score(y_test, yhat_test, zero_division=0)
        try:
            auc = roc_auc_score(y_test, yproba_test)
        except Exception:
            auc = float("nan")

        st.subheader("Chỉ tiêu kiểm định tổng quan")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1-score", f"{f1:.3f}")
        m5.metric("ROC-AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")

        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.subheader("Ma trận nhầm lẫn")
            cm = confusion_matrix(y_test, yhat_test)
            cm_df = pd.DataFrame(
                cm,
                index=[f"Thực tế: {TARGET_LABELS[0]}", f"Thực tế: {TARGET_LABELS[1]}"],
                columns=[f"Dự đoán: {TARGET_LABELS[0]}", f"Dự đoán: {TARGET_LABELS[1]}"],
            )
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                x=[f"Dự đoán: {TARGET_LABELS[0]}", f"Dự đoán: {TARGET_LABELS[1]}"],
                y=[f"Thực tế: {TARGET_LABELS[0]}", f"Thực tế: {TARGET_LABELS[1]}"],
            )
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_roc:
            st.subheader("Đường cong ROC")
            fpr, tpr, _ = roc_curve(y_test, yproba_test)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
            fig_roc.update_layout(
                height=380, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                margin=dict(t=30),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        st.subheader("Báo cáo phân loại chi tiết (classification report)")
        report_dict = classification_report(
            y_test, yhat_test,
            target_names=[TARGET_LABELS[0], TARGET_LABELS[1]],
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

# ------------------------------------------------------------
# TAB 6: SỬ DỤNG MÔ HÌNH
# ------------------------------------------------------------
with tab_use:
    if "model" not in st.session_state:
        st.info("ℹ️ Chưa có mô hình nào được huấn luyện. Vui lòng bấm nút '🚀 Huấn luyện mô hình' ở thanh bên trái.")
    else:
        model = st.session_state["model"]
        feature_cols = st.session_state["results"]["feature_cols"]

        mode = st.radio(
            "Chọn chế độ sử dụng",
            options=["Nhập trực tiếp", "Tải file hàng loạt"],
            horizontal=True,
        )

        if mode == "Nhập trực tiếp":
            st.caption("Nhập điểm đánh giá (1-5) cho từng tiêu chí 5C của khách hàng.")
            with st.form("predict_form"):
                input_values = {}
                for group_name, cols in FEATURE_GROUPS.items():
                    st.markdown(f"**{group_name}**")
                    grp_cols = st.columns(len(cols))
                    for gcol, cname in zip(grp_cols, cols):
                        default_val = int(round(df[cname].median()))
                        with gcol:
                            input_values[cname] = st.number_input(
                                cname,
                                min_value=int(df[cname].min()),
                                max_value=int(df[cname].max()),
                                value=default_val,
                                step=1,
                                key=f"input_{cname}",
                            )
                submitted = st.form_submit_button("Dự báo", type="primary", use_container_width=True)

            if submitted:
                X_new = pd.DataFrame([[input_values[c] for c in feature_cols]], columns=feature_cols)
                pred = model.predict(X_new)[0]
                proba = model.predict_proba(X_new)[0]

                label = TARGET_LABELS[int(pred)]
                if pred == 1:
                    st.error(f"⚠️ Kết quả dự báo: **{label}**")
                else:
                    st.success(f"✅ Kết quả dự báo: **{label}**")

                p1, p2 = st.columns(2)
                p1.metric("Xác suất không có rủi ro", f"{proba[0]*100:.2f}%")
                p2.metric("Xác suất có rủi ro", f"{proba[1]*100:.2f}%")

        else:
            st.caption(
                "Tải lên file CSV chứa đúng các cột biến đầu vào (TC1..TS4, 24 cột) "
                "để dự báo hàng loạt."
            )
            batch_file = st.file_uploader("Tải file dữ liệu khách hàng mới (.csv)", type=["csv"], key="batch_upload")

            if batch_file is not None:
                try:
                    new_df = pd.read_csv(batch_file)
                except Exception:
                    batch_file.seek(0)
                    new_df = pd.read_csv(batch_file, encoding="cp1258")

                missing_batch_cols = [c for c in feature_cols if c not in new_df.columns]
                if missing_batch_cols:
                    st.error(f"❌ File thiếu các cột bắt buộc: {', '.join(missing_batch_cols)}")
                else:
                    X_batch = new_df[feature_cols]
                    preds = model.predict(X_batch)
                    probas = model.predict_proba(X_batch)[:, 1]

                    result_df = new_df.copy()
                    result_df["Dự_báo_PD"] = preds
                    result_df["Nhãn_dự_báo"] = result_df["Dự_báo_PD"].map(TARGET_LABELS)
                    result_df["Xác_suất_rủi_ro (%)"] = (probas * 100).round(2)

                    st.subheader("Kết quả dự báo hàng loạt")
                    with st.container(height=400):
                        st.dataframe(result_df, use_container_width=True)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ Tải kết quả (CSV)",
                        data=csv_bytes,
                        file_name="ket_qua_du_bao_5C.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
