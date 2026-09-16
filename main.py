import math

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")

st.title("🎬 영화 유형 나누기")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


df_raw = load_data()
total_count = len(df_raw)

# ---------- 전처리 ----------
df = df_raw.copy()

required_cols = ["movieNm", "first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
df = df.dropna(subset=required_cols)
df = df[(df["first_week_audi"] != 0) & (df["first_scrn"] > 0) & (df["total_audi"] > 0)]

df["log_first_scrn"] = df["first_scrn"].apply(math.log10)
df["log_total_audi"] = df["total_audi"].apply(math.log10)
df["longrun_index"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)

used_count = len(df)

st.write(f"전체 {total_count}편 중 {used_count}편을 묶는 데 사용했습니다.")

# ---------- 속성 선택 ----------
FEATURE_OPTIONS = {
    "스크린 수 (로그)": "log_first_scrn",
    "누적 관객 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "longrun_index",
}

st.subheader("묶는 데 사용할 속성 선택")
selected_labels = st.multiselect(
    "속성을 둘 이상 골라 주세요.",
    options=list(FEATURE_OPTIONS.keys()),
    default=list(FEATURE_OPTIONS.keys()),
)

if len(selected_labels) < 2:
    st.warning("속성을 둘 이상 선택해 주세요.")
    st.stop()

selected_cols = [FEATURE_OPTIONS[label] for label in selected_labels]

# ---------- 표준화 + k-평균 ----------
X = df[selected_cols].values
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 큰 묶음부터 ㉮, ㉯, ㉰
order = df.groupby("cluster_raw")["total_audi"].mean().sort_values(ascending=False).index.tolist()
name_map = {cid: name for cid, name in zip(order, ["㉮", "㉯", "㉰"])}
df["cluster"] = df["cluster_raw"].map(name_map)
cluster_order = ["㉮", "㉯", "㉰"]

# ---------- 2차원 산점도 ----------
st.subheader("2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_label_2d = st.selectbox("가로축", selected_labels, index=0, key="x2d")
with col2:
    y_label_2d = st.selectbox(
        "세로축", selected_labels, index=1 if len(selected_labels) > 1 else 0, key="y2d"
    )

x_col_2d = FEATURE_OPTIONS[x_label_2d]
y_col_2d = FEATURE_OPTIONS[y_label_2d]

fig2d = px.scatter(
    df,
    x=x_col_2d,
    y=y_col_2d,
    color="cluster",
    category_orders={"cluster": cluster_order},
    hover_name="movieNm",
    labels={x_col_2d: x_label_2d, y_col_2d: y_label_2d},
)
st.plotly_chart(fig2d, use_container_width=True)

# ---------- 3차원 산점도 ----------
st.subheader("3차원 산점도")
if len(selected_labels) < 3:
    st.info("3차원 산점도를 그리려면 속성을 셋 이상 선택해 주세요.")
else:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_label_3d = st.selectbox("X축", selected_labels, index=0, key="x3d")
    with col4:
        y_label_3d = st.selectbox("Y축", selected_labels, index=1, key="y3d")
    with col5:
        z_label_3d = st.selectbox("Z축", selected_labels, index=2, key="z3d")

    x_col_3d = FEATURE_OPTIONS[x_label_3d]
    y_col_3d = FEATURE_OPTIONS[y_label_3d]
    z_col_3d = FEATURE_OPTIONS[z_label_3d]

    fig3d = px.scatter_3d(
        df,
        x=x_col_3d,
        y=y_col_3d,
        z=z_col_3d,
        color="cluster",
        category_orders={"cluster": cluster_order},
        hover_name="movieNm",
        labels={x_col_3d: x_label_3d, y_col_3d: y_label_3d, z_col_3d: z_label_3d},
    )
    fig3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig3d, use_container_width=True)

# ---------- 묶음별 요약 표 ----------
st.subheader("묶음별 요약")
summary = (
    df.groupby("cluster")
    .agg(
        편수=("movieNm", "count"),
        스크린_수_평균=("first_scrn", "mean"),
        누적_관객_평균=("total_audi", "mean"),
        십위권_일수_평균=("days_in_top10", "mean"),
        롱런_지수_평균=("longrun_index", "mean"),
    )
    .reindex(cluster_order)
)
st.dataframe(summary)

# ---------- 묶음별 누적 관객 상위 5편 ----------
st.subheader("묶음별 누적 관객 상위 5편")
for c in cluster_order:
    top5 = df[df["cluster"] == c].sort_values("total_audi", ascending=False).head(5)
    st.write(f"**{c}**: " + ", ".join(top5["movieNm"].tolist()))
