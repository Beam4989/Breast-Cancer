# app_streamlit.py
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ================= CONFIG =================
COLS = [
    "clump_thickness",
    "uniform_cell_size",
    "uniform_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
]
ART_DIR = Path("artifacts")

# ================= TEXT MULTILINGUAL =================
TEXT = {
    "th": {
        "title": "🔎 เครื่องมือช่วยตัดสินใจมะเร็งเต้านม",
        "caption": "เพื่อการศึกษาเท่านั้น (ไม่ใช่การวินิจฉัยทางการแพทย์)",
        "desc": (
            "แบบจำลองนี้ใช้ข้อมูลเซลล์จากชุดข้อมูล Wisconsin Breast Cancer "
            "เพื่อช่วยคาดเดาว่าตัวอย่างอาจเป็นมะเร็งหรือไม่ "
            "โดยผู้ใช้สามารถกรอกค่าลักษณะของเซลล์ (1–10) "
            "แล้วระบบจะประเมินผลความเป็นไปได้เบื้องต้น"
        ),
        "input_title": "🧬 กรอกค่าคุณลักษณะของเซลล์ (1–10)",
        "submit": "ทำนายผล",
        "result_pos": "🩸 ผลลัพธ์: เป็นมะเร็งเต้านม (Positive)",
        "result_neg": "🟢 ผลลัพธ์: ไม่เป็นมะเร็งเต้านม (Negative)",
        "note": "หมายเหตุ: เป็นเพียงเครื่องมือช่วยตัดสินใจเบื้องต้น ไม่ใช่การวินิจฉัยโดยแพทย์",
        "lang_label": "ภาษา",
        "scale_hint": "สเกล 1 = ปกติ / ต่ำ ... 10 = ผิดปกติ / สูง",
    },
    "en": {
        "title": "🔎 Breast Cancer – Decision Support Tool",
        "caption": "For educational use only (not a medical diagnosis).",
        "desc": (
            "This model uses data from the Wisconsin Breast Cancer dataset "
            "to help predict whether a cell sample may be malignant or benign. "
            "Enter feature values (1–10) to get a preliminary prediction."
        ),
        "input_title": "🧬 Enter feature values (scale 1–10)",
        "submit": "Predict Result",
        "result_pos": "🩸 Result: Positive (Breast cancer likely)",
        "result_neg": "🟢 Result: Negative (No breast cancer)",
        "note": "Note: Decision support only – not a medical diagnosis.",
        "lang_label": "Language",
        "scale_hint": "Scale 1 = normal/low … 10 = abnormal/high",
    },
}

# ================= FEATURE DETAILS =================
FEATURES = [
    {
        "key": "clump_thickness",
        "th": {
            "name": "ความหนาแน่นของกลุ่มเซลล์",
            "desc": "บ่งบอกความหนาแน่นของกลุ่มเซลล์ หากหนามากอาจสื่อถึงการเจริญเติบโตผิดปกติ",
        },
        "en": {
            "name": "Clump Thickness",
            "desc": "Describes how densely the cells are grouped. High thickness may suggest abnormal growth.",
        },
    },
    {
        "key": "uniform_cell_size",
        "th": {
            "name": "ขนาดเซลล์สม่ำเสมอ",
            "desc": "ถ้าเซลล์มีขนาดแตกต่างกันมากอาจเป็นลักษณะของมะเร็ง",
        },
        "en": {
            "name": "Uniform Cell Size",
            "desc": "Indicates how similar the sizes of the cells are. Large variation may indicate cancer.",
        },
    },
    {
        "key": "uniform_cell_shape",
        "th": {
            "name": "รูปร่างเซลล์สม่ำเสมอ",
            "desc": "เซลล์ปกติควรมีรูปร่างคล้ายกัน ความไม่สม่ำเสมออาจบ่งชี้ถึงการกลายพันธุ์",
        },
        "en": {
            "name": "Uniform Cell Shape",
            "desc": "Normal cells have consistent shapes. Irregular shapes may indicate mutation or malignancy.",
        },
    },
    {
        "key": "marginal_adhesion",
        "th": {
            "name": "การยึดเกาะของขอบเซลล์",
            "desc": "ดูว่าขอบเซลล์ยึดเกาะกันดีแค่ไหน เซลล์มะเร็งมักหลุดออกง่าย",
        },
        "en": {
            "name": "Marginal Adhesion",
            "desc": "Measures how strongly cells adhere at the edges. Weak adhesion can indicate cancer cells.",
        },
    },
    {
        "key": "single_epithelial_cell_size",
        "th": {
            "name": "ขนาดเซลล์เยื่อบุเดี่ยว",
            "desc": "เซลล์เยื่อบุเดี่ยวที่ใหญ่ผิดปกติอาจเป็นสัญญาณของมะเร็ง",
        },
        "en": {
            "name": "Single Epithelial Cell Size",
            "desc": "Large isolated epithelial cells often appear in malignant tissue.",
        },
    },
    {
        "key": "bare_nuclei",
        "th": {
            "name": "นิวเคลียสเปลือย",
            "desc": "จำนวนเซลล์ที่เห็นนิวเคลียสชัดเจน (ไม่มีไซโทพลาซึม) มักเพิ่มขึ้นในเซลล์มะเร็ง",
        },
        "en": {
            "name": "Bare Nuclei",
            "desc": "Number of cells with clearly visible nuclei without cytoplasm; often higher in cancer.",
        },
    },
    {
        "key": "bland_chromatin",
        "th": {
            "name": "โครมาตินเรียบ",
            "desc": "โครงสร้างของโครมาตินในนิวเคลียส หากไม่เรียบหรือหนาแน่นมากอาจบ่งถึงเซลล์มะเร็ง",
        },
        "en": {
            "name": "Bland Chromatin",
            "desc": "Texture of chromatin in the nucleus. Coarse or irregular texture suggests malignancy.",
        },
    },
    {
        "key": "normal_nucleoli",
        "th": {
            "name": "นิวคลีโอลี",
            "desc": "เซลล์มะเร็งมักมีนิวคลีโอลีที่เด่นชัดหรือมีจำนวนมาก",
        },
        "en": {
            "name": "Normal Nucleoli",
            "desc": "Nucleoli prominence – multiple or large nucleoli often seen in malignant cells.",
        },
    },
    {
        "key": "mitoses",
        "th": {
            "name": "ไมโทซิส (การแบ่งเซลล์)",
            "desc": "บ่งบอกอัตราการแบ่งตัวของเซลล์ การแบ่งตัวมากผิดปกติเป็นสัญญาณของมะเร็ง",
        },
        "en": {
            "name": "Mitoses",
            "desc": "Indicates the rate of cell division. High mitotic count suggests cancerous activity.",
        },
    },
]

# ================= LOAD MODEL =================
@st.cache_resource
def load_artifacts():
    pre = joblib.load(ART_DIR / "preprocessor.pkl")
    model = joblib.load(ART_DIR / "best_model.pkl")
    return pre, model

# ================= UI =================
st.set_page_config(page_title="Breast Cancer Decision Support", layout="wide")

# Sidebar: Language selection
lang = st.sidebar.radio(
    label=TEXT["en"]["lang_label"] + " / " + TEXT["th"]["lang_label"],
    options=["th", "en"],
    format_func=lambda x: "ไทย" if x == "th" else "English",
    index=0,
)
T = TEXT[lang]

# Header
st.title(T["title"])
st.caption(T["caption"])
st.write(T["desc"])
st.divider()

pre, model = load_artifacts()

st.subheader(T["input_title"])
st.write("📏 " + T["scale_hint"])

vals = {}
cols_per_row = 3

# ======= Feature Input Section =======
for i in range(0, len(FEATURES), cols_per_row):
    row_feats = FEATURES[i : i + cols_per_row]
    col_objs = st.columns(len(row_feats))
    for f, col in zip(row_feats, col_objs):
        f_info = f[lang]
        with col:
            vals[f["key"]] = st.number_input(
                label=f_info["name"],
                min_value=1,
                max_value=10,
                value=3,
                step=1,
            )
            st.caption(f"🧠 {f_info['desc']}")

st.divider()

# ======= Prediction Section =======
if st.button(T["submit"], use_container_width=True):
    X = pd.DataFrame([vals], columns=COLS)
    X_s = pre.transform(X)
    proba = model.predict_proba(X_s)[:, 1][0]
    pred = proba >= 0.5

    st.subheader("🩺 ผลการประเมิน / Prediction Result")
    if pred:
        st.error(T["result_pos"])
    else:
        st.success(T["result_neg"])
    st.caption(T["note"])
