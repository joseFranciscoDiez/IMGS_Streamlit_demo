# app.py
# ---------------------------------------------------------
# Demo Streamlit: Segmentación simple sobre "coins" (skimage)
# - Binariza con Otsu, limpia máscara, etiqueta regiones.
# - Muestra métricas por región: media, mediana, min, max, área, bbox.
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st
from skimage import data, filters, morphology, measure, color

# ---------------- Configuración de la página ----------------
st.set_page_config(
    page_title="Monedas (skimage): binarización y regiones",
    page_icon="🪙",
    layout="centered",
)

st.title("Monedas de skimage: binarización y regiones")
st.write("Ajusta los parámetros en la barra lateral. Los resultados se actualizan al instante.")

# ---------------- Barra lateral (parámetros) ----------------
st.sidebar.header("Parámetros")
min_obj = st.sidebar.slider("Tamaño mínimo de objetos (px)", 0, 500, 50, 5)
min_hole = st.sidebar.slider("Tamaño mínimo de agujeros (px)", 0, 500, 50, 5)
st.sidebar.caption("Valores mayores eliminan más ruido y rellenan más huecos.")

# ---------------- Carga y procesamiento ----------------
# 1) Carga imagen ejemplo (grayscale)
img = data.coins()  # ndarray 2D (H, W), uint8

# 2) Umbral de Otsu y máscara binaria
threshold = filters.threshold_otsu(img)
mask = img > threshold  # bool

# 3) Limpieza morfológica
#    - quita componentes pequeñas
#    - rellena agujeros pequeños
mask_clean = morphology.remove_small_objects(mask, min_size=min_obj)
mask_clean = morphology.remove_small_holes(mask_clean, area_threshold=min_hole)

# 4) Etiquetado de regiones
label_img = measure.label(mask_clean, connectivity=1)

# 5) Cálculo de métricas por región
props = measure.regionprops(label_img, intensity_image=img)
rows = []
for r in props:
    # Intensidades sólo dentro de la región (en su recorte)
    intensities = r.intensity_image[r.image]
    median_val = float(np.median(intensities)) if intensities.size > 0 else float("nan")

    rows.append({
        "label": int(r.label),
        "area": int(r.area),
        "bbox_r0": int(r.bbox[0]),
        "bbox_c0": int(r.bbox[1]),
        "bbox_r1": int(r.bbox[2]),
        "bbox_c1": int(r.bbox[3]),
        "mean": float(r.mean_intensity),
        "median": median_val,
        "min": float(r.min_intensity),
        "max": float(r.max_intensity),
    })

df = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
df_display = df.copy()
for col in ["mean", "median", "min", "max"]:
    if col in df_display.columns:
        df_display[col] = df_display[col].round(2)

# 6) Visualizaciones
overlay = color.label2rgb(label_img, image=img, bg_label=0, alpha=0.3)

col1, col2, col3 = st.columns(3)
with col1:
    st.image(img, caption="Imagen original (grises)", use_column_width=True)
with col2:
    st.image(mask, caption=f"Máscara binaria (Otsu={threshold})", use_column_width=True)
with col3:
    st.image(overlay, caption=f"Etiquetas sobrepuestas (regiones: {label_img.max()})", use_column_width=True)

st.subheader("Características por región")
st.dataframe(df_display, use_container_width=True)

st.caption(
    "Notas: "
    "bbox = (r0, c0, r1, c1) en coordenadas de imagen. "
    "Las intensidades son las del canal en escala de grises."
)
