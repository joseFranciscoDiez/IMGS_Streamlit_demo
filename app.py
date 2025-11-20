# app.py
# ---------------------------------------------------------
# Clase práctica: Streamlit + scikit-image (coins)
# - Binariza con Otsu, limpia máscara, etiqueta regiones.
# - Muestra métricas por región y permite descargar CSV.
# ---------------------------------------------------------

import time
import numpy as np
import pandas as pd
import streamlit as st
from skimage import data, filters, morphology, measure, color

# 1) Configuración general de la página
st.set_page_config(
    page_title="Streamlit + skimage: Coins demo",
    page_icon="🪙",
    layout="wide",  # más espacio para columnas
)

# 2) Título e instrucciones breves
st.title("🪙 Segmentación simple con skimage + Streamlit")
st.write("Esta app binariza la imagen de monedas, limpia la máscara, etiqueta regiones y calcula métricas.")

# 3) Barra lateral: parámetros de usuario (widgets)
st.sidebar.header("Parámetros")
min_obj = st.sidebar.slider("Tamaño mínimo de objetos (px)", 0, 1000, 50, 10)
min_hole = st.sidebar.slider("Tamaño mínimo de agujeros (px)", 0, 1000, 50, 10)

# Botón opcional para limpiar caché (demostración)
# Nota: si aún no se definió compute_pipeline en esta ejecución, lo indicamos.
if st.sidebar.button("🧹 Limpiar caché de datos"):
    try:
        compute_pipeline.clear()
        st.sidebar.success("Caché limpiada.")
    except NameError:
        st.sidebar.info("La función cacheada aún no se definió en esta sesión.")

# 4) Carga de la imagen (coins)
img = data.coins()  # 2D uint8 (H, W), escala de grises
H, W = img.shape

# 5) Pipeline de procesamiento cacheado (st.cache_data)
@st.cache_data(show_spinner=False)
def compute_pipeline(img: np.ndarray, min_obj: int, min_hole: int):
    # 5.1) Umbral de Otsu
    threshold = filters.threshold_otsu(img)
    mask = img > threshold  # bool

    # 5.2) Limpieza morfológica (condicional para permitir 0)
    mask_clean = morphology.remove_small_objects(mask, min_size=min_obj) if min_obj > 0 else mask
    mask_clean = morphology.remove_small_holes(mask_clean, area_threshold=min_hole) if min_hole > 0 else mask_clean

    # 5.3) Etiquetado (conectividad por defecto de skimage)
    label_img = measure.label(mask_clean)

    # 5.4) Propiedades por región
    props = measure.regionprops(label_img, intensity_image=img)
    rows = []
    for r in props:
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

    # 5.5) Overlay de etiquetas sobre la imagen original
    overlay = color.label2rgb(label_img, image=img, bg_label=0, alpha=0.3)
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    return {
        "threshold": float(threshold),
        "mask": mask,
        "mask_clean": mask_clean,
        "label_img": label_img,
        "df": df,
        "overlay": overlay_uint8,
    }

# 6) Ejecutar pipeline (con spinner y medición de tiempo)
with st.spinner("Procesando..."):
    t0 = time.time()
    out = compute_pipeline(img, min_obj, min_hole)
    dt = time.time() - t0

# 7) Métricas rápidas y visualizaciones
c_top1, c_top2, c_top3 = st.columns(3)
with c_top1:
    st.metric("Umbral Otsu", f"{out['threshold']:.1f}")
with c_top2:
    st.metric("Regiones detectadas", int(out["label_img"].max()))
with c_top3:
    st.metric("Tiempo (s)", f"{dt:.3f}")

# Tres columnas con imágenes clave
col1, col2, col3 = st.columns(3)
with col1:
    st.image(img, caption=f"Original ({W}x{H})", use_column_width=True, clamp=True)
with col2:
    st.image(out["mask_clean"], caption="Máscara limpia", use_column_width=True)
with col3:
    st.image(out["overlay"], caption="Etiquetas sobrepuestas", use_column_width=True)

# 8) Tabla de características y descarga
st.subheader("Características por región")
df_display = out["df"].copy()
for col in ["mean", "median", "min", "max"]:
    if col in df_display.columns:
        df_display[col] = df_display[col].round(2)
st.dataframe(df_display, use_container_width=True)

csv_bytes = out["df"].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar tabla (CSV)", data=csv_bytes, file_name="region_props.csv", mime="text/csv")

# 9) Explicación breve (útil para clase; se puede ocultar en un expander)
with st.expander("¿Qué está pasando aquí?"):
    st.markdown(
        "- Streamlit ejecuta este script de arriba a abajo cada vez que cambias un slider o haces clic.\n"
        "- Los sliders en la barra lateral controlan parámetros del procesamiento.\n"
        "- st.cache_data guarda el resultado de compute_pipeline para evitar recomputar si no cambian los parámetros.\n"
        "- Mostramos imágenes y una tabla con st.image y st.dataframe.\n"
        "- st.download_button permite exportar resultados (CSV).\n"
    )
