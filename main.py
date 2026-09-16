import math

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")

st.title("🎬 영화 유형 나누기")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

# 묶음 수에 따라 붙일 기호 (최대 7개)
CLUSTER_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]


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

# ---------- 묶음 수 선택 ----------
st.subheader("묶음 수 선택")
n_clusters = st.slider("묶음 수를 골라 주세요.", min_value=2, max_value=7, value=3)

cluster_symbols = CLUSTER_SYMBOLS[:n_clusters]

# ---------- 표준화 + k-평균 ----------
X = df[selected_cols].values
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 큰 묶음부터 순서대로 기호 부여
order = df.groupby("cluster_raw")["total_audi"].mean().sort_values(ascending=False).index.tolist()
name_map = {cid: name for cid, name in zip(order, cluster_symbols)}
df["cluster"] = df["cluster_raw"].map(name_map)
cluster_order = cluster_symbols

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

# ---------- 묶음 수에 따른 변화 (엘보우) ----------
st.subheader("묶음 수에 따른 변화")

k_range = list(range(1, 8))
inertias = []
for k in k_range:
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_k.fit(X_scaled)
    inertias.append(km_k.inertia_)

fig_elbow = go.Figure()
fig_elbow.add_trace(
    go.Scatter(x=k_range, y=inertias, mode="lines+markers", name="묶음 내 거리 제곱합")
)
fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="red")
fig_elbow.update_layout(
    xaxis_title="묶음 수",
    yaxis_title="묶음 내 거리 제곱합",
    xaxis=dict(dtick=1),
)
st.plotly_chart(fig_elbow, use_container_width=True)

elbow_table = pd.DataFrame({"묶음 수": k_range, "거리 제곱합": inertias})
elbow_table["직전 대비 감소량"] = (-elbow_table["거리 제곱합"].diff()).round(3)
elbow_table.loc[elbow_table.index[0], "직전 대비 감소량"] = pd.NA
st.dataframe(elbow_table.set_index("묶음 수"))

# ---------- 실루엣 점수 ----------
sil_score = silhouette_score(X_scaled, df["cluster_raw"])
st.write(f"**묶음 수 {n_clusters}개의 실루엣 점수: {sil_score:.3f}** (-1~1, 1에 가까울수록 묶음이 뚜렷함)")
