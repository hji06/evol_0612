import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="하디-바인베르크 탐구 시뮬레이터", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.3rem;padding-bottom:2rem}
.hero{padding:1.35rem 1.5rem;border-radius:18px;color:white;background:linear-gradient(135deg,#0f172a,#1e3a5f);margin-bottom:1rem}
.hero h1{margin:0;font-size:1.9rem}.hero p{margin:.4rem 0 0;color:#dbeafe}
.box{padding:1rem 1.1rem;border-radius:14px;background:#f8fafc;border-left:5px solid #2563eb;margin:.8rem 0}
.warn{padding:1rem 1.1rem;border-radius:14px;background:#fff7ed;border-left:5px solid #f97316;margin:.8rem 0}
.ok{padding:1rem 1.1rem;border-radius:14px;background:#f0fdf4;border-left:5px solid #16a34a;margin:.8rem 0}
.pred{padding:1rem 1.1rem;border-radius:14px;background:#eef2ff;border-left:5px solid #6366f1;margin:.8rem 0}
.small{color:#64748b;font-size:.92rem}
</style>
""", unsafe_allow_html=True)

TOPICS = [
    "① 평형 상태 살펴보기",
    "② 집단 크기와 우연의 영향",
    "③ 자연선택의 영향",
    "④ 돌연변이의 영향",
    "⑤ 집단 사이 이동의 영향",
    "⑥ 짝짓기 방식의 영향",
]

# 각 주제가 '하디-바인베르크 평형의 5가지 조건' 중 무엇을 깨는지 명시
BROKEN = {
    "① 평형 상태 살펴보기": "다섯 가지 평형 조건이 모두 충족된 이상적인 상태 (어떤 조건도 깨지 않음)",
    "② 집단 크기와 우연의 영향": "[큰 집단] 조건 — 집단이 충분히 커야 우연의 영향이 작아진다",
    "③ 자연선택의 영향": "[자연선택 없음] 조건 — 모든 유전자형의 생존·번식 확률이 같아야 한다",
    "④ 돌연변이의 영향": "[돌연변이 없음] 조건 — 대립유전자가 다른 것으로 바뀌지 않아야 한다",
    "⑤ 집단 사이 이동의 영향": "[집단 간 이동 없음] 조건 — 다른 집단과 개체 교류가 없어야 한다",
    "⑥ 짝짓기 방식의 영향": "[자유로운 짝짓기] 조건 — 모든 개체가 무작위로 짝짓기해야 한다",
}

GUIDE = {
    "① 평형 상태 살펴보기": {
        "q": "A 대립유전자의 시작 비율이 달라지면 AA, Aa, aa 비율은 어떻게 달라질까?",
        "changed": "A 대립유전자 시작 비율",
        "fixed": "평형 조건, 세대 수",
        "checks": ["A 시작 비율이 다른 두 조건을 비교한다.", "AA, Aa, aa 중 어떤 유전자형의 비율이 큰지 확인한다.", "세대가 지나도 그래프가 변하는지 확인한다."],
        "questions": ["조건 A와 조건 B에서 가장 많이 나타나는 유전자형은 각각 무엇인가?", "그래프가 세대에 따라 거의 변하지 않는다면, 그 이유를 평형 조건과 연결해 설명해 보자."],
    },
    "② 집단 크기와 우연의 영향": {
        "q": "집단 크기가 작을수록 A 대립유전자 비율은 더 크게 흔들릴까?",
        "changed": "집단 크기",
        "fixed": "A 대립유전자 시작 비율, 세대 수, 반복 횟수",
        "checks": ["조건 A와 조건 B의 집단 크기만 다르게 설정한다.", "반복 실험 선들이 얼마나 넓게 퍼지는지 비교한다.", "최종 세대에서 A 비율의 최솟값과 최댓값을 확인한다."],
        "questions": ["두 조건 중 반복 실험 결과가 더 넓게 퍼진 조건은 무엇인가?", "같은 조건으로 여러 번 반복했는데 결과가 서로 다른 이유를 추론해 보자."],
    },
    "③ 자연선택의 영향": {
        "q": "생존과 번식에 불리한 정도가 커지면 대립유전자 비율은 더 빠르게 변할까?",
        "changed": "불리한 정도",
        "fixed": "A 대립유전자 시작 비율, Aa가 불리함을 받는 정도, 세대 수",
        "checks": ["조건 A와 조건 B의 불리한 정도만 다르게 설정한다.", "A와 a의 비율이 어느 방향으로 변하는지 확인한다.", "변화가 빠른 조건과 느린 조건을 그래프 기울기로 비교한다."],
        "questions": ["두 조건 중 대립유전자 비율이 더 빠르게 변한 조건은 무엇인가?", "자연선택이 작용하면 집단의 유전적 구성이 왜 달라질 수 있는지 설명해 보자."],
    },
    "④ 돌연변이의 영향": {
        "q": "A에서 a로 바뀌는 비율이 커지면 A 대립유전자 비율은 어떻게 변할까?",
        "changed": "A에서 a로 바뀌는 비율",
        "fixed": "A 대립유전자 시작 비율, a에서 A로 바뀌는 비율, 세대 수",
        "checks": ["조건 A와 조건 B에서 A에서 a로 바뀌는 비율만 다르게 설정한다.", "변화 방향과 변화 속도를 함께 확인한다.", "최종 세대의 A 비율 차이를 표에서 확인한다."],
        "questions": ["A에서 a로 바뀌는 비율이 큰 조건에서 A 대립유전자 비율은 어떻게 변했는가?", "돌연변이만으로 집단의 비율이 크게 바뀌려면 어떤 조건이 필요할지 생각해 보자."],
    },
    "⑤ 집단 사이 이동의 영향": {
        "q": "다른 집단에서 개체가 많이 들어오면 원래 집단의 대립유전자 비율은 어떻게 변할까?",
        "changed": "들어오는 개체의 비율",
        "fixed": "A 대립유전자 시작 비율, 들어오는 집단의 A 비율, 세대 수",
        "checks": ["조건 A와 조건 B에서 들어오는 개체의 비율만 다르게 설정한다.", "원래 집단의 A 비율이 어느 값 쪽으로 가까워지는지 확인한다.", "두 집단의 차이가 커지는지 줄어드는지 판단한다."],
        "questions": ["들어오는 개체의 비율이 큰 조건에서 원래 집단의 A 비율은 어떻게 변했는가?", "집단 사이 이동이 계속되면 두 집단의 유전적 차이는 어떻게 될지 추론해 보자."],
    },
    "⑥ 짝짓기 방식의 영향": {
        "q": "가까운 개체끼리 짝짓기하는 정도가 커지면 Aa 비율은 어떻게 변할까?",
        "changed": "가까운 개체끼리 짝짓기하는 정도",
        "fixed": "A 대립유전자 시작 비율, 세대 수",
        "checks": ["조건 A와 조건 B에서 짝짓기 방식만 다르게 설정한다.", "A와 a의 비율이 변하는지 먼저 확인한다.", "AA, Aa, aa 중 어느 비율이 가장 크게 변하는지 확인한다."],
        "questions": ["가까운 개체끼리 짝짓기하는 정도가 큰 조건에서 Aa 비율은 어떻게 변했는가?", "대립유전자 비율과 유전자형 비율이 항상 같이 변하는지 그래프를 근거로 판단해 보자."],
    },
}

COLOR = {"A": "#2563eb", "B": "#ea580c", "AA": "#0f766e", "Aa": "#7c3aed", "aa": "#dc2626"}

# ---------------------------------------------------------------------------
# 모델 (점화식)
# ---------------------------------------------------------------------------

def geno(p):
    """무작위 교배 가정에서 갓 태어난 자손(접합자)의 유전자형 비율 (p², 2pq, q²)."""
    q = 1 - p
    return p * p, 2 * p * q, q * q

def make_rows(p_values):
    rows = []
    for g, p in enumerate(p_values):
        AA, Aa, aa = geno(p)
        rows.append({"세대": g, "A 비율": p, "a 비율": 1 - p, "AA": AA, "Aa": Aa, "aa": aa})
    return pd.DataFrame(rows)

def sim_basic(p0, gen):
    return make_rows([p0] * (gen + 1))

def sim_selection(p0, bad, h, gen):
    p, vals = p0, []
    for _ in range(gen + 1):
        vals.append(p)
        q = 1 - p
        w_AA, w_Aa, w_aa = 1, 1 - h * bad, 1 - bad
        w_bar = p**2 * w_AA + 2 * p * q * w_Aa + q**2 * w_aa
        if w_bar <= 0:
            p = 0
        else:
            p = (p**2 * w_AA + p * q * w_Aa) / w_bar
        p = min(max(p, 0), 1)
    return make_rows(vals)

def sim_mutation(p0, A_to_a, a_to_A, gen):
    p, vals = p0, []
    for _ in range(gen + 1):
        vals.append(p)
        p = p * (1 - A_to_a) + (1 - p) * a_to_A
        p = min(max(p, 0), 1)
    return make_rows(vals)

def sim_migration(p0, incoming_A, incoming_rate, gen):
    p, vals = p0, []
    for _ in range(gen + 1):
        vals.append(p)
        p = (1 - incoming_rate) * p + incoming_rate * incoming_A
        p = min(max(p, 0), 1)
    return make_rows(vals)

def sim_mating(p0, close_degree, gen):
    """근친교배: 대립유전자 비율은 유지되고 이형접합(Aa)만 (1-F)^g로 감소."""
    q = 1 - p0
    H0 = 2 * p0 * q
    rows = []
    for g in range(gen + 1):
        Aa = H0 * (1 - close_degree) ** g
        AA = p0 - Aa / 2
        aa = q - Aa / 2
        rows.append({"세대": g, "A 비율": p0, "a 비율": q, "AA": max(AA, 0), "Aa": max(Aa, 0), "aa": max(aa, 0)})
    return pd.DataFrame(rows)

def sim_drift(p0, N, gen, repeats, number):
    rng = np.random.default_rng(number)
    rows = []
    for r in range(1, repeats + 1):
        p = p0
        for g in range(gen + 1):
            rows.append({"반복": r, "세대": g, "A 비율": p, "a 비율": 1 - p})
            if g < gen and 0 < p < 1:
                p = rng.binomial(2 * N, p) / (2 * N)
    df = pd.DataFrame(rows)
    summary = df.groupby("세대").agg(평균_A=("A 비율", "mean"), 최소_A=("A 비율", "min"), 최대_A=("A 비율", "max")).reset_index()
    return df, summary

def sim_by_topic(topic, p):
    if topic == "① 평형 상태 살펴보기":
        return sim_basic(p["start"], p["gen"])
    if topic == "③ 자연선택의 영향":
        return sim_selection(p["start"], p["bad"], p["h"], p["gen"])
    if topic == "④ 돌연변이의 영향":
        return sim_mutation(p["start"], p["A_to_a"], p["a_to_A"], p["gen"])
    if topic == "⑤ 집단 사이 이동의 영향":
        return sim_migration(p["start"], p["incoming_A"], p["incoming_rate"], p["gen"])
    if topic == "⑥ 짝짓기 방식의 영향":
        return sim_mating(p["start"], p["close"], p["gen"])
    raise ValueError("지원하지 않는 주제입니다.")

def equilibrium_value(topic, p):
    """이론적으로 수렴하는 A 비율(평형값). 없으면 None."""
    if topic == "④ 돌연변이의 영향":
        denom = p["A_to_a"] + p["a_to_A"]
        if denom > 0:
            return p["a_to_A"] / denom
        return None
    if topic == "⑤ 집단 사이 이동의 영향":
        if p["incoming_rate"] > 0:
            return p["incoming_A"]
        return None
    return None

# ---------------------------------------------------------------------------
# 그래프
# ---------------------------------------------------------------------------

def allele_fig(a, b, title, hlines=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a["세대"], y=a["A 비율"], name="조건 A: A", mode="lines", line=dict(color=COLOR["A"], width=3)))
    fig.add_trace(go.Scatter(x=b["세대"], y=b["A 비율"], name="조건 B: A", mode="lines", line=dict(color=COLOR["B"], width=3)))
    fig.add_trace(go.Scatter(x=a["세대"], y=a["a 비율"], name="조건 A: a", mode="lines", line=dict(color=COLOR["A"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=b["세대"], y=b["a 비율"], name="조건 B: a", mode="lines", line=dict(color=COLOR["B"], width=2, dash="dot")))
    # 이론적 평형값(수렴값) 기준선
    if hlines:
        for y, color, label in hlines:
            if y is None:
                continue
            fig.add_hline(y=y, line=dict(color=color, width=1.5, dash="dash"),
                          annotation_text=label, annotation_position="right",
                          annotation_font=dict(color=color, size=11))
    fig.update_layout(title=title, xaxis_title="세대", yaxis_title="대립유전자 비율", yaxis=dict(range=[0, 1]), height=420, legend=dict(orientation="h", y=1.15))
    return fig

def genotype_fig(a, b, title):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("조건 A", "조건 B"))
    for col, df in [(1, a), (2, b)]:
        fig.add_trace(go.Scatter(x=df["세대"], y=df["AA"], name="AA", mode="lines", line=dict(color=COLOR["AA"], width=2.5), showlegend=(col == 1)), row=1, col=col)
        fig.add_trace(go.Scatter(x=df["세대"], y=df["Aa"], name="Aa", mode="lines", line=dict(color=COLOR["Aa"], width=2.5), showlegend=(col == 1)), row=1, col=col)
        fig.add_trace(go.Scatter(x=df["세대"], y=df["aa"], name="aa", mode="lines", line=dict(color=COLOR["aa"], width=2.5), showlegend=(col == 1)), row=1, col=col)
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(title=title, height=420, legend=dict(orientation="h", y=1.15))
    return fig

def drift_fig(run_a, sum_a, run_b, sum_b):
    fig = go.Figure()
    for r, part in run_a.groupby("반복"):
        fig.add_trace(go.Scatter(x=part["세대"], y=part["A 비율"], name="조건 A 반복", mode="lines", line=dict(color=COLOR["A"], width=1), opacity=.22, showlegend=(r == 1)))
    for r, part in run_b.groupby("반복"):
        fig.add_trace(go.Scatter(x=part["세대"], y=part["A 비율"], name="조건 B 반복", mode="lines", line=dict(color=COLOR["B"], width=1), opacity=.22, showlegend=(r == 1)))
    fig.add_trace(go.Scatter(x=sum_a["세대"], y=sum_a["평균_A"], name="조건 A 평균", mode="lines", line=dict(color=COLOR["A"], width=4)))
    fig.add_trace(go.Scatter(x=sum_b["세대"], y=sum_b["평균_A"], name="조건 B 평균", mode="lines", line=dict(color=COLOR["B"], width=4)))
    fig.update_layout(title="집단 크기에 따른 우연의 영향 비교", xaxis_title="세대", yaxis_title="A 대립유전자 비율", yaxis=dict(range=[0, 1]), height=450, legend=dict(orientation="h", y=1.16))
    return fig

# ---------------------------------------------------------------------------
# 표/문자열 유틸
# ---------------------------------------------------------------------------

def param_text(p):
    labels = {
        "gen": "세대 수", "start": "처음 A 비율", "N": "집단 크기", "repeats": "반복 횟수",
        "bad": "불리한 정도", "h": "Aa가 불리함을 받는 정도",
        "A_to_a": "A→a 변화 비율", "a_to_A": "a→A 변화 비율",
        "incoming_A": "들어오는 집단의 A 비율", "incoming_rate": "들어오는 개체의 비율",
        "close": "가까운 개체끼리 짝짓기 정도"
    }
    return ", ".join(f"{labels.get(k, k)}={v}" for k, v in p.items() if k != "gen")

def result_row(label, df, p):
    first, last = df.iloc[0], df.iloc[-1]
    return {
        "조건": label,
        "처음 A 비율": round(float(first["A 비율"]), 4),
        "마지막 A 비율": round(float(last["A 비율"]), 4),
        "A 비율 변화량": round(float(last["A 비율"] - first["A 비율"]), 4),
        "마지막 a 비율": round(float(last["a 비율"]), 4),
        "마지막 AA": round(float(last["AA"]), 4),
        "마지막 Aa": round(float(last["Aa"]), 4),
        "마지막 aa": round(float(last["aa"]), 4),
        "조건값": param_text(p),
    }

def drift_row(label, df, p):
    final = df[df["세대"] == df["세대"].max()]
    return {
        "조건": label,
        "처음 A 비율": round(float(p["start"]), 4),
        "마지막 평균 A 비율": round(float(final["A 비율"].mean()), 4),
        "마지막 A 비율 최솟값": round(float(final["A 비율"].min()), 4),
        "마지막 A 비율 최댓값": round(float(final["A 비율"].max()), 4),
        "A 비율이 1이 된 횟수": int((final["A 비율"] >= 1).sum()),
        "A 비율이 0이 된 횟수": int((final["A 비율"] <= 0).sum()),
        "조건값": param_text(p),
    }

def summarize_for_record(summary_df):
    """결과 요약표를 한 줄 문자열로 압축(탐구 기록 자동 첨부용)."""
    parts = []
    for _, row in summary_df.iterrows():
        kv = ", ".join(f"{c}={row[c]}" for c in summary_df.columns if c not in ("조건", "조건값"))
        parts.append(f"[{row['조건']}] {kv}")
    return " / ".join(parts)

def to_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")

# ---------------------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------------------

if "records" not in st.session_state:
    st.session_state.records = []
if "show" not in st.session_state:
    st.session_state.show = False
if "prev_topic" not in st.session_state:
    st.session_state.prev_topic = None

st.markdown("""
<div class="hero">
<h1>하디-바인베르크 법칙 탐구 시뮬레이터</h1>
<p>평형 조건이 깨질 때 집단의 유전적 구성이 어떻게 변하는지 조건 A와 조건 B로 비교해 봅니다.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("탐구 설정")
    topic = st.selectbox("오늘 탐구할 내용", TOPICS)
    gen = st.slider("몇 세대까지 볼까요?", 10, 200, 60, 10)

    if topic == "① 평형 상태 살펴보기":
        common_start = 0.5
    else:
        common_start = st.slider("처음 A 대립유전자 비율", 0.01, 0.99, 0.50, 0.01)
        st.caption(f"처음 a 대립유전자 비율: {1 - common_start:.2f}")

    # 시드(반복 실험 번호)는 우연이 들어가는 ②번에서만 노출
    if topic == "② 집단 크기와 우연의 영향":
        experiment_no = st.number_input(
            "반복 실험 번호",
            min_value=1, max_value=99999, value=2026, step=1,
            help="우연이 들어가는 실험에서 결과를 다시 똑같이 재현하기 위한 번호입니다. 번호를 바꾸면 다른 우연 결과를 볼 수 있습니다.",
        )
    else:
        experiment_no = 2026

# 주제를 바꾸면 결과 화면을 초기화 → 학생이 예상을 먼저 쓰도록 유도
if st.session_state.prev_topic is not None and st.session_state.prev_topic != topic:
    st.session_state.show = False
st.session_state.prev_topic = topic

# ---------------------------------------------------------------------------
# 1. 탐구 질문과 예상
# ---------------------------------------------------------------------------

st.subheader("1. 탐구 질문과 예상 세우기")
st.markdown(f"""
<div class="box">
<b>이번 주제에서 깨는 평형 조건</b><br>{BROKEN[topic]}<br><br>
<b>탐구 질문</b><br>{GUIDE[topic]["q"]}<br><br>
<span class="small">그래프를 보기 전에 먼저 예상해 보세요. 예상이 틀려도 괜찮습니다.</span>
</div>
""", unsafe_allow_html=True)

with st.form("plan"):
    research_q = st.text_input("나의 탐구 질문", value=GUIDE[topic]["q"])
    prediction = st.text_area("그래프를 보기 전 나의 예상", placeholder="정답을 쓰는 칸이 아니라, 내 생각을 먼저 적는 칸입니다.")
    reason = st.text_area("그렇게 예상한 까닭", placeholder="왜 그렇게 생각했는지 개념, 직관, 이전 경험을 바탕으로 적어 보세요.")
    st.form_submit_button("예상 저장하기")

if not prediction.strip():
    st.markdown("<div class='warn'><b>수업 안내</b><br>학생이 결과를 보기 전에 예상부터 쓰도록 안내하면 탐구 활동으로 활용하기 좋습니다.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. 조건 A / 조건 B 설정
# ---------------------------------------------------------------------------

st.subheader("2. 조건 A와 조건 B 설정하기")
st.markdown(f"""
<div class="box">
<b>다르게 할 조건</b>: {GUIDE[topic]["changed"]}<br>
<b>같게 유지할 조건</b>: {GUIDE[topic]["fixed"]}<br>
<span class="small">가능하면 한 가지 조건만 다르게 설정하세요.</span>
</div>
""", unsafe_allow_html=True)

a_col, b_col = st.columns(2)
pa, pb = {"gen": gen}, {"gen": gen}

if topic == "① 평형 상태 살펴보기":
    with a_col:
        st.markdown("#### 조건 A")
        pa["start"] = st.slider("조건 A: 처음 A 비율", 0.01, 0.99, 0.30, 0.01, key="basic_a")
    with b_col:
        st.markdown("#### 조건 B")
        pb["start"] = st.slider("조건 B: 처음 A 비율", 0.01, 0.99, 0.70, 0.01, key="basic_b")

elif topic == "② 집단 크기와 우연의 영향":
    repeats = st.slider("반복 실험 횟수", 3, 30, 10, 1)
    with a_col:
        st.markdown("#### 조건 A")
        pa.update({"start": common_start, "N": st.slider("조건 A: 집단 크기", 5, 1000, 50, 5, key="N_a"), "repeats": repeats})
    with b_col:
        st.markdown("#### 조건 B")
        pb.update({"start": common_start, "N": st.slider("조건 B: 집단 크기", 5, 1000, 500, 5, key="N_b"), "repeats": repeats})

elif topic == "③ 자연선택의 영향":
    h = st.slider("공통 조건: Aa가 불리함을 받는 정도", 0.0, 1.0, 0.5, 0.05)
    with a_col:
        st.markdown("#### 조건 A")
        pa.update({"start": common_start, "h": h, "bad": st.slider("조건 A: aa가 생존·번식에 불리한 정도", 0.0, 1.0, 0.10, 0.01, key="bad_a")})
    with b_col:
        st.markdown("#### 조건 B")
        pb.update({"start": common_start, "h": h, "bad": st.slider("조건 B: aa가 생존·번식에 불리한 정도", 0.0, 1.0, 0.40, 0.01, key="bad_b")})

elif topic == "④ 돌연변이의 영향":
    a_to_A = st.slider("공통 조건: a에서 A로 바뀌는 비율", 0.0, 0.05, 0.001, 0.0005, format="%.4f")
    with a_col:
        st.markdown("#### 조건 A")
        pa.update({"start": common_start, "a_to_A": a_to_A, "A_to_a": st.slider("조건 A: A에서 a로 바뀌는 비율", 0.0, 0.05, 0.002, 0.0005, format="%.4f", key="mut_a")})
    with b_col:
        st.markdown("#### 조건 B")
        pb.update({"start": common_start, "a_to_A": a_to_A, "A_to_a": st.slider("조건 B: A에서 a로 바뀌는 비율", 0.0, 0.05, 0.020, 0.0005, format="%.4f", key="mut_b")})

elif topic == "⑤ 집단 사이 이동의 영향":
    incoming_A = st.slider("공통 조건: 들어오는 집단의 A 비율", 0.01, 0.99, 0.80, 0.01)
    with a_col:
        st.markdown("#### 조건 A")
        pa.update({"start": common_start, "incoming_A": incoming_A, "incoming_rate": st.slider("조건 A: 들어오는 개체의 비율", 0.0, 0.5, 0.05, 0.01, key="move_a")})
    with b_col:
        st.markdown("#### 조건 B")
        pb.update({"start": common_start, "incoming_A": incoming_A, "incoming_rate": st.slider("조건 B: 들어오는 개체의 비율", 0.0, 0.5, 0.20, 0.01, key="move_b")})

elif topic == "⑥ 짝짓기 방식의 영향":
    with a_col:
        st.markdown("#### 조건 A")
        pa.update({"start": common_start, "close": st.slider("조건 A: 가까운 개체끼리 짝짓기하는 정도", 0.0, 1.0, 0.10, 0.05, key="close_a")})
    with b_col:
        st.markdown("#### 조건 B")
        pb.update({"start": common_start, "close": st.slider("조건 B: 가까운 개체끼리 짝짓기하는 정도", 0.0, 1.0, 0.50, 0.05, key="close_b")})

with st.expander("현재 조건 표로 확인하기"):
    st.dataframe(pd.DataFrame([{"조건": "A", **pa}, {"조건": "B", **pb}]), use_container_width=True, hide_index=True)

st.markdown("""
<div class="ok">
<b>관찰 방법</b><br>
조건 A와 조건 B에서 무엇이 다른지 확인한 뒤, A 비율, a 비율, AA·Aa·aa 비율 중 무엇이 변했는지 찾아보세요.
</div>
""", unsafe_allow_html=True)

if st.button("시뮬레이션 실행하기", type="primary"):
    st.session_state.show = True

if not st.session_state.show:
    st.info("조건을 설정한 뒤 '시뮬레이션 실행하기' 버튼을 누르세요.")

# ---------------------------------------------------------------------------
# 3. 결과 관찰
# ---------------------------------------------------------------------------

if st.session_state.show:
    st.subheader("3. 결과 관찰하기")

    # 예상-관찰-설명(POE): 결과를 보기 직전, 내가 쓴 예상을 다시 보여 줌
    if prediction.strip():
        reason_html = f"<br><span class='small'>예상한 까닭: {reason.strip()}</span>" if reason.strip() else ""
        st.markdown(f"""
<div class="pred">
<b>그래프를 보기 전 내가 적은 예상</b><br>{prediction.strip()}{reason_html}<br><br>
<span class="small">아래 결과와 내 예상이 같은지 다른지 비교하면서 보세요.</span>
</div>
""", unsafe_allow_html=True)

    if topic == "② 집단 크기와 우연의 영향":
        run_a, sum_a = sim_drift(pa["start"], pa["N"], gen, pa["repeats"], int(experiment_no))
        run_b, sum_b = sim_drift(pb["start"], pb["N"], gen, pb["repeats"], int(experiment_no) + 777)
        st.plotly_chart(drift_fig(run_a, sum_a, run_b, sum_b), use_container_width=True)
        st.caption("얇은 선은 각 반복 실험, 굵은 선은 평균입니다. 집단이 작을수록 선이 넓게 흩어집니다. (조건 A·B에 서로 다른 반복 실험 번호를 적용해 우연 흐름을 다르게 했습니다.)")
        summary = pd.DataFrame([drift_row("A", run_a, pa), drift_row("B", run_b, pb)])
    else:
        df_a, df_b = sim_by_topic(topic, pa), sim_by_topic(topic, pb)
        if topic == "① 평형 상태 살펴보기":
            st.plotly_chart(genotype_fig(df_a, df_b, "유전자형 비율 비교"), use_container_width=True)
            st.caption("매 세대 무작위 교배를 가정한 자손(접합자) 기준 비율입니다. 평형 상태에서는 세대가 지나도 비율이 변하지 않습니다.")
        elif topic == "⑥ 짝짓기 방식의 영향":
            left, right = st.columns(2)
            with left:
                st.plotly_chart(allele_fig(df_a, df_b, "대립유전자 비율 변화"), use_container_width=True)
            with right:
                st.plotly_chart(genotype_fig(df_a, df_b, "유전자형 비율 변화"), use_container_width=True)
            st.caption("대립유전자(A·a) 비율은 그대로지만 유전자형(AA·Aa·aa) 비율은 변할 수 있다는 점에 주목하세요.")
        else:
            hlines = [
                (equilibrium_value(topic, pa), COLOR["A"], "조건 A 수렴값"),
                (equilibrium_value(topic, pb), COLOR["B"], "조건 B 수렴값"),
            ]
            st.plotly_chart(allele_fig(df_a, df_b, "대립유전자 비율 변화", hlines=hlines), use_container_width=True)
            if any(y is not None for y, _, _ in hlines):
                st.caption("점선 가로선은 이론적으로 수렴하는 A 비율(평형값)입니다. 곡선이 이 값으로 가까워지는지 확인해 보세요.")
            st.caption("AA·Aa·aa 비율은 매 세대 무작위 교배를 가정한 자손(접합자) 기준입니다. 작은 집단에서는 선택·돌연변이도 우연으로 흔들릴 수 있으나, 여기서는 요인을 분리해 보기 위해 매끈한 곡선으로 보여 줍니다.")
            with st.expander("AA, Aa, aa 비율 그래프도 보기"):
                st.plotly_chart(genotype_fig(df_a, df_b, "유전자형 비율 변화"), use_container_width=True)
        summary = pd.DataFrame([result_row("A", df_a, pa), result_row("B", df_b, pb)])

    st.markdown("#### 결과 요약표")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button("결과 요약표 내려받기", data=to_csv(summary), file_name="hardy_weinberg_result_summary.csv", mime="text/csv")

    # -----------------------------------------------------------------------
    # 5. 자료 해석 질문 (기록 저장보다 먼저 렌더 → 한 번에 같이 저장)
    # -----------------------------------------------------------------------
    st.subheader("4. 자료 해석 질문")
    st.markdown("""
<div class="warn">
<b>주의</b><br>
아래 질문에는 예시 답안을 보여주지 않습니다. 자신이 실행한 조건 A, 조건 B의 그래프와 표를 근거로 직접 작성해 보세요.
</div>
""", unsafe_allow_html=True)

    q_answers = []
    for i, q in enumerate(GUIDE[topic]["questions"], start=1):
        ans = st.text_area(f"질문 {i}. {q}", placeholder="그래프와 표에서 확인한 근거를 포함해 자신의 말로 작성하세요.", key=f"q_{topic}_{i}")
        q_answers.append((q, ans))

    with st.expander("답을 쓰기 어려울 때 보는 도움말"):
        st.markdown("- 먼저 조건 A와 조건 B에서 다르게 한 조건을 찾습니다.")
        st.markdown("- 그래프에서 위아래로 크게 변한 선이 있는지 봅니다.")
        st.markdown("- 결과 요약표에서 마지막 값과 변화량을 확인합니다.")
        st.markdown("- 문장은 `조건 A는 ..., 조건 B는 ...`처럼 비교해서 씁니다.")
        st.markdown("- 마지막에는 `따라서 나는 ...라고 판단했다`처럼 자신의 결론을 씁니다.")

    # -----------------------------------------------------------------------
    # 5. 관찰 기록 작성 (조건값·결과·질문 답안을 함께 저장)
    # -----------------------------------------------------------------------
    st.subheader("5. 관찰 기록 작성하기")
    with st.expander("기록할 때 확인할 점 보기"):
        for point in GUIDE[topic]["checks"]:
            st.markdown(f"- {point}")
        st.markdown("- 그래프의 모양만 쓰지 말고, 표의 숫자도 함께 확인합니다.")
        st.markdown("- 조건 A와 조건 B를 비교하는 문장으로 작성합니다.")
        st.markdown("- 예상이 틀려도 괜찮습니다. 중요한 것은 근거를 들어 설명하는 것입니다.")

    with st.form("record"):
        changed = st.text_input("이번 비교에서 다르게 한 조건", value=GUIDE[topic]["changed"])
        fixed = st.text_input("같게 유지한 조건", value=GUIDE[topic]["fixed"])
        observation = st.text_area("그래프에서 관찰한 사실", placeholder="조건 A와 조건 B를 비교해서, 어떤 값이 어떻게 달라졌는지 적어 보세요.")
        evidence = st.text_area("표나 그래프에서 찾은 근거", placeholder="마지막 세대 값, 변화량, 최솟값·최댓값 등 숫자를 포함해 적어 보세요.")
        interpretation = st.text_area("나의 해석과 결론", placeholder="관찰한 결과가 왜 나타났는지 생명과학 개념과 연결해 설명해 보세요.")
        saved = st.form_submit_button("탐구 기록 저장하기 (예상·조건·결과·질문 답안 함께 저장)")

    if saved:
        record = {
            "탐구 주제": topic,
            "깨는 평형 조건": BROKEN[topic],
            "탐구 질문": research_q,
            "그래프 보기 전 예상": prediction,
            "예상한 까닭": reason,
            "다르게 한 조건": changed,
            "같게 유지한 조건": fixed,
            "조건 A 설정값": param_text(pa),
            "조건 B 설정값": param_text(pb),
            "결과 요약(자동)": summarize_for_record(summary),
            "관찰한 사실": observation,
            "근거": evidence,
            "해석과 결론": interpretation,
        }
        # 자료 해석 질문 답안을 함께 저장
        for i, (q, ans) in enumerate(q_answers, start=1):
            record[f"질문{i}"] = q
            record[f"질문{i} 답안"] = ans
        st.session_state.records.append(record)
        st.success("탐구 기록이 저장되었습니다. (예상·조건·결과·질문 답안이 함께 저장되었습니다.)")

    if st.session_state.records:
        st.markdown("#### 저장된 탐구 기록")
        st.markdown("<div class='warn'><b>꼭 확인하세요</b><br>저장된 기록은 새로고침하거나 창을 닫으면 사라집니다. 활동이 끝나면 아래 '탐구 기록 내려받기'로 반드시 파일을 저장하세요.</div>", unsafe_allow_html=True)
        records = pd.DataFrame(st.session_state.records)
        st.dataframe(records, use_container_width=True, hide_index=True)
        st.download_button("탐구 기록 내려받기", data=to_csv(records), file_name="hardy_weinberg_inquiry_records.csv", mime="text/csv")
        if st.button("저장된 탐구 기록 모두 지우기"):
            st.session_state.records = []
            st.rerun()

st.divider()
st.caption("하디-바인베르크 법칙 학생 중심 탐구 수업용 Streamlit 웹앱")
