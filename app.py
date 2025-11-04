# -*- coding: utf-8 -*-
# 웹 버전 포스터 생성기 (한국어 UI)
# 실행: streamlit run app.py

import io
import os
import math
import random
import colorsys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Streamlit 백엔드
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import streamlit as st

APP_TITLE = "인터랙티브 포스터 생성기 (웹)"
PALETTE_CSV_DEFAULT = "palette.csv"

# -----------------------------
# 유틸
# -----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def adjust_color(rgb, contrast=1.0, sat=1.0, bright=0.0, hue_shift=0.0, light_boost=0.0):
    """HSV 기반 색 보정 + 명암(대비) + 조명 부스트."""
    r, g, b = rgb
    r, g, b = clamp01(r), clamp01(g), clamp01(b)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # hue_shift: degree 또는 0..1 turn 모두 허용
    hs = hue_shift / 360.0 if abs(hue_shift) > 1 else hue_shift
    h = (h + hs) % 1.0
    s = clamp01(s * sat)
    v = clamp01(0.5 + (v - 0.5) * contrast + bright + light_boost)

    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (r2, g2, b2)

# -----------------------------
# 셰이프(도형)
# -----------------------------
def blob(center=(0.5, 0.5), r=0.3, points=240, wobble=0.15, phase_shift=0.0):
    """유기적 블롭 좌표 생성."""
    angles = np.linspace(0, 2 * math.pi, points, endpoint=False)
    angles = angles + phase_shift * np.sin(3 * angles)
    radii = r * (1 + wobble * (np.random.rand(points) - 0.5))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return x, y

def gen_blob_polygon(center, r, points, wobble, phase_shift):
    x, y = blob(center=center, r=r, points=points, wobble=wobble, phase_shift=phase_shift)
    return Polygon(np.column_stack((x, y)), closed=True)

def gen_block_rect(x, y, w, h):
    return Rectangle((x, y), w, h)

# -----------------------------
# 팔레트 로딩 & 생성기
# -----------------------------
def _gen_pastel(n):
    return [(random.uniform(0.4, 0.9),
             random.uniform(0.4, 0.9),
             random.uniform(0.4, 0.9)) for _ in range(n)]

def _gen_vivid(n):
    cols = []
    for _ in range(n):
        hi = random.randint(0, 2)
        c = [random.uniform(0.0, 0.5) for _ in range(3)]
        c[hi] = random.uniform(0.7, 1.0)
        cols.append(tuple(c))
    return cols

def _gen_mono(n, base=None):
    if base is None:
        base = (random.random(), random.random(), random.random())
    br, bg, bb = base
    cols = []
    for i in range(n):
        f = 0.4 + 0.6 * (i / max(1, n - 1))
        cols.append((br * f, bg * f, bb * f))
    return cols

def _gen_random(n):
    return [(random.random(), random.random(), random.random()) for _ in range(n)]

def _normalize_palette_rows(vals):
    out = []
    for r, g, b in vals:
        r, g, b = float(r), float(g), float(b)
        if max(r, g, b) > 1.0:  # 0..255 스케일을 0..1로 정규화
            r, g, b = r / 255.0, g / 255.0, b / 255.0
        out.append((clamp01(r), clamp01(g), clamp01(b)))
    return out

def load_palette_from_df(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    if all(k in cols for k in ("r", "g", "b")) and len(df) > 0:
        vals = df[[cols["r"], cols["g"], cols["b"]]].values.tolist()
        return _normalize_palette_rows(vals)
    return None

def load_palette_from_csv(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            pal = load_palette_from_df(df)
            if pal:
                return pal
        except Exception:
            pass
    return None

def get_palette_by_mode(mode: str, n_colors: int, seed: int, uploaded_df: pd.DataFrame | None):
    random.seed(seed); np.random.seed(seed)
    m = (mode or "").lower()

    if m == "csv":
        # 업로드 우선, 없으면 로컬 파일, 그래도 없으면 파스텔 기본
        if uploaded_df is not None:
            pal = load_palette_from_df(uploaded_df)
            if pal: return pal
        pal = load_palette_from_csv(PALETTE_CSV_DEFAULT)
        if pal: return pal
        return _gen_pastel(max(4, n_colors))

    if m == "pastel":  return _gen_pastel(n_colors)
    if m == "vivid":   return _gen_vivid(n_colors)
    if m == "mono":    return _gen_mono(n_colors)
    if m == "random":  return _gen_random(n_colors)
    return _gen_pastel(n_colors)

# -----------------------------
# 그리기
# -----------------------------
def draw_poster_plus(
    n_layers=8,
    wobble=0.15,
    palette_mode="pastel",
    seed=0,
    points=260,
    bg_color="#ffffff",
    margin=0.03,
    contrast=1.0,
    saturation=1.0,
    brightness=0.0,
    hue_shift_deg=0.0,
    palette_rotate=0,
    palette_shuffle=False,
    enable_shadow=True,
    shadow_layers=3,
    shadow_offset=0.02,
    shadow_spread=0.012,
    shadow_alpha=0.25,
    light_boost=0.0,
    show_edges=False,
    edge_width=1.0,
    edge_alpha=0.6,
    edge_color="#000000",
    global_alpha=0.68,
    randomize_order=True,
    shape_mode="blob",
    block_rows=4,
    block_cols=4,
    block_jitter=0.35,
    block_size_min=0.5,
    uploaded_df: pd.DataFrame | None = None,
):
    random.seed(seed); np.random.seed(seed)

    total_needed = max(n_layers, block_rows * block_cols)
    palette = get_palette_by_mode(palette_mode, total_needed, seed, uploaded_df)

    # 팔레트 회전/섞기
    if len(palette) > 0 and palette_rotate:
        k = int(palette_rotate) % len(palette)
        palette = palette[k:] + palette[:k]
    if palette_shuffle:
        random.shuffle(palette)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor(bg_color)

    patches = []
    colors = []

    if shape_mode == "blob":
        layer_indices = list(range(int(n_layers)))
        if randomize_order:
            random.shuffle(layer_indices)
        for idx in layer_indices:
            cx = random.uniform(margin, 1 - margin)
            cy = random.uniform(margin, 1 - margin)
            r = random.uniform(0.18, 0.32) * (1 - 0.2 * random.random())
            phase = random.uniform(0.0, 1.0)
            poly = gen_blob_polygon(center=(cx, cy), r=r, points=int(points),
                                    wobble=wobble * (1 + 0.12 * idx), phase_shift=phase)
            base_col = palette[idx % len(palette)]
            lb = light_boost * (idx / max(1, n_layers - 1))
            adj = adjust_color(base_col, contrast, saturation, brightness, hue_shift_deg, lb)
            patches.append(poly)
            colors.append(adj)
    else:
        rows, cols = int(block_rows), int(block_cols)
        cell_w = (1 - 2 * margin) / cols
        cell_h = (1 - 2 * margin) / rows
        ids = [(r, c) for r in range(rows) for c in range(cols)]
        if randomize_order:
            random.shuffle(ids)
        k = 0
        for r, c in ids:
            cx = margin + c * cell_w + cell_w / 2
            cy = margin + r * cell_h + cell_h / 2
            jx = (random.random() * 2 - 1) * block_jitter * cell_w / 2
            jy = (random.random() * 2 - 1) * block_jitter * cell_h / 2
            size_scale = block_size_min + (1 - block_size_min) * random.random()
            w = cell_w * size_scale
            h = cell_h * size_scale
            x = cx - w / 2 + jx
            y = cy - h / 2 + jy
            rect = gen_block_rect(x, y, w, h)
            base_col = palette[k % len(palette)]
            k += 1
            lb = light_boost * (k / max(1, rows * cols - 1))
            adj = adjust_color(base_col, contrast, saturation, brightness, hue_shift_deg, lb)
            patches.append(rect)
            colors.append(adj)

    # 그림자(쉐도우)
    if enable_shadow and shadow_layers > 0:
        shadow_patches = []
        shadow_colors = []
        for p in patches:
            for s in range(int(shadow_layers)):
                off = shadow_offset + s * shadow_spread
                dx = off * 0.8
                dy = -off
                if isinstance(p, Polygon):
                    verts = p.get_xy()
                    sp = Polygon(verts + np.array([dx, dy]), closed=True)
                elif isinstance(p, Rectangle):
                    sp = Rectangle((p.get_x() + dx, p.get_y() + dy), p.get_width(), p.get_height())
                else:
                    continue
                a = clamp01(shadow_alpha * (1 - s / max(1, shadow_layers)))
                shadow_patches.append(sp)
                shadow_colors.append((0, 0, 0, a))
        shadow_coll = PatchCollection(shadow_patches, match_original=False)
        shadow_coll.set_facecolor(shadow_colors)
        shadow_coll.set_edgecolor("none")
        shadow_coll.set_zorder(0)
        ax.add_collection(shadow_coll)

    # 전경 도형
    coll = PatchCollection(patches, alpha=global_alpha, match_original=False)
    coll.set_facecolor(colors)
    if show_edges:
        coll.set_edgecolor(edge_color)
        coll.set_linewidth(edge_width)
        # 별도의 엣지 컬렉션으로 독립 알파
        edge_coll = PatchCollection(patches, facecolor="none", edgecolor=edge_color,
                                    linewidth=edge_width, alpha=edge_alpha)
        edge_coll.set_zorder(3)
        ax.add_collection(edge_coll)
    else:
        coll.set_edgecolor("none")
    coll.set_zorder(2)
    ax.add_collection(coll)

    # 축 설정
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    title_mode = f"{palette_mode} • {shape_mode}"
    ax.set_title(f"Poster • {title_mode}", loc="left", fontsize=16, fontweight="bold")

    return fig

# -----------------------------
# Streamlit UI
# -----------------------------
def parse_uploaded_palette(file) -> pd.DataFrame | None:
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
        return df
    except Exception:
        try:
            # 탭/세미콜론 등 변형도 시도
            df = pd.read_csv(file, sep=None, engine="python")
            return df
        except Exception:
            return None

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("CSV 팔레트, 3D(그림자), 색상 대비/채도/명도, 블롭/색상 블록을 모두 웹에서 조절할 수 있습니다.")

    with st.sidebar:
        st.header("팔레트 불러오기")
        up = st.file_uploader("CSV 업로드 (R,G,B 열 필요 / 0..1 또는 0..255)", type=["csv"])
        uploaded_df = parse_uploaded_palette(up)

        st.header("기본 설정")
        seed = st.number_input("시드(Seed)", min_value=0, max_value=999999, value=0, step=1)
        palette_mode = st.selectbox("팔레트 모드", ["pastel", "vivid", "mono", "random", "csv"], index=0)
        palette_rotate = st.number_input("팔레트 회전(개수)", min_value=0, max_value=999, value=0, step=1)
        palette_shuffle = st.checkbox("팔레트 셔플", value=False)
        bg_color = st.color_picker("배경색", "#ffffff")
        margin = st.slider("여백(Margin)", 0.0, 0.2, 0.03, 0.005)

        st.header("색상 보정")
        contrast = st.slider("대비(Contrast)", 0.2, 2.5, 1.0, 0.05)
        saturation = st.slider("채도(Saturation)", 0.0, 2.0, 1.0, 0.05)
        brightness = st.slider("명도(Brightness)", -0.5, 0.5, 0.0, 0.02)
        hue_shift_deg = st.slider("색상 회전(Hue, °)", -180, 180, 0, 5)

        st.header("그림자/3D")
        enable_shadow = st.checkbox("그림자 사용", value=True)
        shadow_layers = st.slider("그림자 레이어 수", 1, 8, 3, 1)
        shadow_offset = st.slider("그림자 오프셋", 0.0, 0.08, 0.02, 0.002)
        shadow_spread = st.slider("그림자 확산", 0.0, 0.05, 0.012, 0.002)
        shadow_alpha = st.slider("그림자 투명도", 0.0, 0.8, 0.25, 0.02)
        light_boost = st.slider("레이어 조명 부스트", 0.0, 0.5, 0.0, 0.02)

        st.header("외곽선/투명도")
        show_edges = st.checkbox("외곽선 표시", value=False)
        edge_width = st.slider("외곽선 두께", 0.2, 6.0, 1.0, 0.2)
        edge_alpha = st.slider("외곽선 투명도", 0.0, 1.0, 0.6, 0.05)
        edge_color = st.color_picker("외곽선 색", "#000000")
        global_alpha = st.slider("전역 알파(도형 투명도)", 0.1, 1.0, 0.68, 0.02)
        randomize_order = st.checkbox("레이어 순서 랜덤", value=True)

        st.header("도형 모드")
        shape_mode = st.selectbox("모드 선택", ["blob", "blocks"], index=0)

        if shape_mode == "blob":
            n_layers = st.slider("레이어 수", 3, 30, 8, 1)
            wobble = st.slider("블롭 요철(Wobble)", 0.01, 0.6, 0.15, 0.01)
            points = st.slider("블롭 포인트 수", 80, 600, 260, 10)
            block_rows = 4
            block_cols = 4
            block_jitter = 0.35
            block_size_min = 0.5
        else:
            n_layers = 8  # 블록 모드에선 rows*cols가 실질 레이어 수 역할
            wobble = 0.15
            points = 260
            col1, col2 = st.columns(2)
            with col1:
                block_rows = st.slider("블록 행", 2, 12, 4, 1)
                block_jitter = st.slider("블록 위치 흔들림", 0.0, 1.0, 0.35, 0.02)
            with col2:
                block_cols = st.slider("블록 열", 2, 12, 4, 1)
                block_size_min = st.slider("블록 최소 크기 비율", 0.2, 1.0, 0.5, 0.02)

    # 그리기
    fig = draw_poster_plus(
        n_layers=n_layers,
        wobble=wobble,
        palette_mode=palette_mode,
        seed=int(seed),
        points=int(points),
        bg_color=bg_color,
        margin=float(margin),
        contrast=float(contrast),
        saturation=float(saturation),
        brightness=float(brightness),
        hue_shift_deg=float(hue_shift_deg),
        palette_rotate=int(palette_rotate),
        palette_shuffle=bool(palette_shuffle),
        enable_shadow=bool(enable_shadow),
        shadow_layers=int(shadow_layers),
        shadow_offset=float(shadow_offset),
        shadow_spread=float(shadow_spread),
        shadow_alpha=float(shadow_alpha),
        light_boost=float(light_boost),
        show_edges=bool(show_edges),
        edge_width=float(edge_width),
        edge_alpha=float(edge_alpha),
        edge_color=edge_color,
        global_alpha=float(global_alpha),
        randomize_order=bool(randomize_order),
        shape_mode=shape_mode,
        block_rows=int(block_rows),
        block_cols=int(block_cols),
        block_jitter=float(block_jitter),
        block_size_min=float(block_size_min),
        uploaded_df=uploaded_df
    )

    st.pyplot(fig, use_container_width=True)

    # 다운로드 버튼 (PNG / SVG)
    colA, colB = st.columns(2)
    with colA:
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05)
        st.download_button("PNG 다운로드", data=buf_png.getvalue(), file_name="poster.png", mime="image/png")
    with colB:
        buf_svg = io.BytesIO()
        fig.savefig(buf_svg, format="svg", bbox_inches="tight", pad_inches=0.05)
        st.download_button("SVG 다운로드", data=buf_svg.getvalue(), file_name="poster.svg", mime="image/svg+xml")

    plt.close(fig)

if __name__ == "__main__":
    # 안전 가드: 예외가 발생하더라도 Streamlit이 깨지지 않도록
    try:
        main()
    except Exception as e:
        # 사용자에게 친절한 에러 박스 (내부 상세는 숨김)
        st.error("알 수 없는 문제가 발생했습니다. 입력 값을 다시 확인해 주세요.")
