"""
Streamlit app for Chapter 4, Exercise 15 - "draw a digit" version.

Run with:
    pip install streamlit streamlit-drawable-canvas scikit-learn pillow pandas numpy
    streamlit run streamlit_digit_draw_app.py
    py -3.14 -m streamlit run streamlit_digit_draw_app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

st.set_page_config(page_title="Draw a digit", layout="wide")


@st.cache_resource
def train_model():
    mnist_df = pd.read_csv("mnist_10k.csv", header=None)
    y = mnist_df.iloc[:, 0].to_numpy().astype(np.uint8)
    X = mnist_df.iloc[:, 1:].to_numpy().astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    return clf, scaler


def canvas_to_mnist_vector(canvas_image: np.ndarray, scaler: StandardScaler) -> tuple[np.ndarray, Image.Image]:
    """Turn the canvas's RGBA drawing into a scaled, MNIST-shaped (1, 784)
    feature vector, plus a small preview image of what will actually be fed
    to the model. Assumes a dark stroke drawn on a light canvas background
    (the app's default settings below)."""
    # The canvas gives RGBA; convert to a single grayscale brightness channel.
    img = Image.fromarray(canvas_image.astype(np.uint8), mode="RGBA").convert("L")

    arr_full = np.array(img)
    ink_mask = arr_full < (arr_full.mean() - arr_full.std() * 0.5)
    if ink_mask.any():
        rows = np.any(ink_mask, axis=1)
        cols = np.any(ink_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Pad to a square canvas (preserving proportions) before resizing to 28x28,
    # the same fix used for the phone-photo pipeline in the notebook.
    w, h = img.size
    pad_frac = 0.7
    side = int(max(w, h) * (1 + pad_frac))
    square = Image.new("L", (side, side), color=255)
    square.paste(img, ((side - w) // 2, (side - h) // 2))
    img = square.resize((28, 28), Image.LANCZOS)

    preview = img.copy()

    arr = np.array(img).astype(np.float64)
    arr = 255.0 - arr  # MNIST = bright digit on dark background
    vec = arr.reshape(1, -1)
    return scaler.transform(vec), preview


st.title("Draw a digit, and the model will recognize it live!")
st.write(
    "Draw a single digit (0-9) below with your mouse. The prediction and the "
    "probability chart update automatically after every stroke -- no button "
    "to click."
)

with st.sidebar:
    st.header("Settings")
    stroke_width = st.slider("Line thickness", 10, 25, 15)
    bg_color = st.color_picker("Background color:", "#FFFFFF")
    stroke_color = st.color_picker("Line color:", "#000000")

col_canvas, col_result = st.columns(2)

with col_canvas:
    st.subheader("Draw here")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=280,
        width=280,
        drawing_mode="freedraw",
        update_streamlit=True,      # sends the drawing back to Streamlit after every stroke
        return_image_data=True,     # required so canvas_result.image_data is populated
        key="digit_canvas",
    )

has_ink = (
    canvas_result.image_data is not None
    and (canvas_result.image_data[:, :, :3] < 250).any()
)

with col_result:
    if not has_ink:
        st.info("Start drawing a digit on the left -- the result will appear here automatically.")
    else:
        clf, scaler = train_model()
        vec_scaled, preview = canvas_to_mnist_vector(canvas_result.image_data, scaler)
        pred = clf.predict(vec_scaled)[0]
        proba = clf.predict_proba(vec_scaled)[0]

        st.subheader("What the model sees (28x28)")
        st.image(preview.resize((140, 140)), clamp=True)

        st.subheader("Prediction")
        st.write(f"**Digit: {pred}**")
        st.write(f"Confidence: {proba[pred]:.2%}")

st.subheader("Probability Distribution")
if has_ink:
    st.bar_chart(pd.Series(proba, index=[str(i) for i in range(10)]))
else:
    st.bar_chart(pd.Series([0.0] * 10, index=[str(i) for i in range(10)]))
