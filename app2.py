# app_spam.py
import streamlit as st
import joblib
import numpy as np
import time

# إعداد الصفحة
st.set_page_config(page_title="📧 مصنّف الإيميلات", page_icon="📧", layout="centered")

# ===== CSS أنيق + نصوص فاتحة بكل العناصر =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
* { font-family: 'Tajawal', sans-serif !important; }

/* خلفية وتدرّج خفيف */
.stApp {
  background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
  background-size: 400% 400%;
  animation: gradientBG 18s ease infinite;
  color: #F8FAFC !important;
}
@keyframes gradientBG {
  0% {background-position:0% 50%}
  50% {background-position:100% 50%}
  100% {background-position:0% 50%}
}

/* كل النصوص */
h1,h2,h3,h4,p,span,li,strong,em { color:#F8FAFC !important; }
label, .stTextArea label, .stTextInput label { color:#F8FAFC !important; opacity:1 !important; }

h1 {
  text-align:center; font-size:38px; font-weight:700;
  color:#60a5fa !important; margin-bottom:20px;
  text-shadow:0 0 14px #3b82f6;
}

/* صندوق الإدخال */
.stTextArea textarea{
  background:#223043 !important;
  color:#ffffff !important;
  border:1px solid #7b8aa0 !important;
  border-radius:12px !important;
  font-size:17px !important; padding:12px !important;
}
.stTextArea textarea::placeholder{
  color:#cfe1ff !important; opacity:1 !important;
}

/* الزر */
.stButton>button{
  background: linear-gradient(90deg, #3b82f6, #06b6d4);
  color:#ffffff !important; border:none; border-radius:25px;
  padding:12px 30px; font-size:18px; font-weight:700; transition:.3s;
}
.stButton>button:hover{ transform:scale(1.05); box-shadow:0 0 20px rgba(59,130,246,.6); }

/* بطاقة النتيجة */
.result-card{
  border-radius:18px; padding:24px; margin-top:25px;
  font-size:22px; font-weight:800; text-align:center;
  animation:fadeInUp 0.9s ease-in-out;
}
.success{ background: linear-gradient(135deg, #10b981, #34d399); color:#ffffff !important; box-shadow:0 0 18px rgba(16,185,129,.35); }
.danger { background: linear-gradient(135deg, #f43f5e, #ef4444); color:#ffffff !important; box-shadow:0 0 18px rgba(239,68,68,.35); }
@keyframes fadeInUp{ from{opacity:0; transform:translateY(25px)} to{opacity:1; transform:translateY(0)} }

/* صناديق التنبيه */
div[role="alert"]{
  background: rgba(148,163,184,0.12) !important;
  color:#F8FAFC !important;
  border:1px solid #7b8aa0 !important;
}
</style>
""", unsafe_allow_html=True)

# ===== تحميل الموديل والفيكتورايزر =====
vectorizer = joblib.load("tfidf_vectorizer_layan.pkl")
model = joblib.load("phishing_model_layan.pkl")

# ===== الواجهة =====
st.markdown("<h1>🚀 مصنّف الإيميلات (Spam / Not Spam)</h1>", unsafe_allow_html=True)

email_text = st.text_area("✍️ أدخل نص الإيميل هنا:", height=180, placeholder="اكتب الإيميل المراد فحصه...")

if st.button("🔮 تصنيف الإيميل"):
    if not email_text.strip():
        st.warning("⚠️ رجاءً اكتب نص الإيميل أولاً")
    else:
        X = vectorizer.transform([email_text])

        # احتمال السبام
        if hasattr(model, "predict_proba"):
            prob_spam = model.predict_proba(X)[0][1]
        else:
            score = model.decision_function(X)
            prob_spam = 1 / (1 + np.exp(-score[0]))

        pred = model.predict(X)[0]  # 1=Spam, 0=Not Spam

        with st.spinner("⏳ جاري تحليل الإيميل..."):
            time.sleep(1.0)

        # النتيجة + سبب
        if pred == 1:
            st.markdown(
                f"<div class='result-card danger'>🚨 Spam<br>نسبة الثقة: {prob_spam*100:.1f}%</div>",
                unsafe_allow_html=True
            )
            st.info("📌 السبب: الموديل لاحظ كلمات أو تراكيب مرتبطة بالرسائل الاحتيالية.")
        else:
            conf_not = (1 - prob_spam) * 100
            st.markdown(
                f"<div class='result-card success'>✅ Not Spam<br>نسبة الثقة: {conf_not:.1f}%</div>",
                unsafe_allow_html=True
            )
            st.info("📌 السبب: النص لا يحتوي مؤشرات قوية للسبام.")

# ===== تذييل =====
st.markdown(
    """
    <hr style="margin-top:30px; margin-bottom:10px; border: 1px solid #334155;" />
    <div style="text-align:center; font-size:14px; color:#94A3B8;">
        👩‍💻 Developed by: <b>Huda, Layan, Rimas, Leena</b>
    </div>
    """,
    unsafe_allow_html=True
)
