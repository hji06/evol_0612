import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="하디-바인베르크 탐구 시뮬레이터",
    page_icon="🧬",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.hero {padding: 1.3rem 1.5rem; border-radius: 16px; color: white;
       background: linear-gradient(135deg, #0f172a, #1e3a5f); margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 1.8rem;}
.hero p {margin: .35rem 0 0; color: #dbeafe;}
.box {padding: 1rem; border-radius: 12px; background: #f8fafc;
      border-left: 5px solid #2563eb; margin: .8rem 0;}
.small {color:#64748b; font-size:.92rem;}
</style>
""", unsafe_allow_html=True)

TOPICS = [
    "① 하디-바인베르크 평형",
    "② 유전적 부동",
    "③ 자연선택",
    "④ 돌연변이",
    "⑤ 유전자 흐름",
    "⑥ 비무작위 교배",
]

GUIDE = {
    "① 하디-바인베르크 평형": {
        "q": "초기 A 대립유전자 빈도 p가 달라지면 유전자형 빈도는 어떻게 달라질까?",
        "changed": "초기 A 대립유전자 빈도 p₀",
        "controlled": "세대 수, 평형 조건 유지",
        "help": "평형 상태에서는 p와 q가 일정하면 p², 2pq, q²도 세대가 지나도 일정하게 유지된다.",
        "quiz": ["p₀가 0.5에 가까울 때 Aa 빈도는 어떻게 나타나는가?", "AA와 aa 빈도가 같아지는 조건은 무엇인가?"],
    },
    "② 유전적 부동": {
        "q": "집단 크기 N이 작아지면 우연에 의한 대립유전자 빈도 변화가 더 커질까?",
        "changed": "집단 크기 N",
        "controlled": "초기 p₀, 세대 수, 반복 횟수",
        "help": "유전적 부동은 우연한 표본 추출 효과이므로 집단 크기가 작을수록 p가 더 크게 흔들리고 고정 또는 소실이 쉽게 일어난다.",
        "quiz": ["A와 B 중 p의 흔들림이 더 큰 조건은 무엇인가?", "같은 조건에서도 반복 실행 결과가 다른 이유는 무엇인가?"],
    },
    "③ 자연선택": {
        "q": "선택 계수 s가 커지면 불리한 대립유전자의 빈도는 더 빠르게 감소할까?",
        "changed": "선택 계수 s",
        "controlled": "초기 p₀, 우세 계수 h, 세대 수",
        "help": "이 앱에서는 AA=1, Aa=1-hs, aa=1-s로 적합도를 둔다. s가 클수록 aa가 더 불리해져 대립유전자 빈도가 더 크게 변한다.",
        "quiz": ["선택 계수가 큰 조건에서 p와 q는 어떻게 변하는가?", "열성 대립유전자가 완전히 사라지기 어려운 이유를 그래프와 연결해 설명하시오."],
    },
    "④ 돌연변이": {
        "q": "A→a 돌연변이율이 커지면 A 대립유전자 빈도 p는 어떻게 변할까?",
        "changed": "전향 돌연변이율 u(A→a)",
        "controlled": "초기 p₀, 역돌연변이율 v, 세대 수",
        "help": "A→a가 a→A보다 크면 p는 감소하는 방향으로 이동한다. 다만 돌연변이만으로는 변화 속도가 대체로 느리다.",
        "quiz": ["돌연변이만으로 p가 빠르게 변했는가, 느리게 변했는가?", "u와 v의 차이가 최종 p에 어떤 영향을 주는가?"],
    },
    "⑤ 유전자 흐름": {
        "q": "이주율 m이 커지면 수용 집단의 p는 이입 집단의 p에 더 빨리 가까워질까?",
        "changed": "이주율 m",
        "controlled": "초기 p₀, 이입 집단의 pₘ, 세대 수",
        "help": "유전자 흐름이 일어나면 수용 집단의 p는 이입 집단의 pₘ 쪽으로 이동한다. m이 클수록 더 빠르게 가까워진다.",
        "quiz": ["수용 집단의 p는 어느 값에 가까워지는가?", "유전자 흐름이 집단 간 차이를 줄이는 이유를 설명하시오."],
    },
    "⑥ 비무작위 교배": {
        "q": "근친교배 계수 F가 커지면 이형접합체 빈도는 어떻게 변할까? p와 q도 변할까?",
        "changed": "근친교배 계수 F",
        "controlled": "초기 p₀, 세대 수",
        "help": "근친교배는 p와 q 자체는 바꾸지 않지만 유전자형 빈도는 바꾼다. Aa는 감소하고 AA와 aa는 증가한다.",
        "quiz": ["F가 큰 조건에서 Aa 빈도는 어떻게 변하는가?", "유전자형 빈도는 변했는데 대립유전자 빈도는 변하지 않는 이유를 설명하시오."],
    },
}

COL = {"A": "#2563eb", "B": "#ea580c", "AA": "#0f766e", "Aa": "#7c3aed", "aa": "#dc2626", "q": "#64748b"}


def hw(p):
    q = 1 - p
    return p * p, 2 * p * q, q * q


def simulate_basic(p0, generations):
    AA, Aa, aa = hw(p0)
    return pd.DataFrame([{"generation": g, "p": p0, "q": 1 - p0, "AA": AA, "Aa": Aa, "aa": aa} for g in range(generations + 1)])


def simulate_selection(p0, s, h, generations):
    p = p0
    rows = []
    for g in range(generations + 1):
        AA, Aa, aa = hw(p)
        rows.append({"generation": g, "p": p, "q": 1 - p, "AA": AA, "Aa": Aa, "aa": aa})
        q = 1 - p
        w_AA, w_Aa, w_aa = 1.0, 1.0 - h * s, 1.0 - s
        w_bar = p**2 * w_AA + 2 * p * q * w_Aa + q**2 * w_aa
        if w_bar <= 0:
            break
        p = (p**2 * w_AA + p * q * w_Aa) / w_bar
        p = min(max(p, 0), 1)
    return pd.DataFrame(rows)


def simulate_mutation(p0, u, v, generations):
    p = p0
    rows = []
    for g in range(generations + 1):
        AA, Aa, aa = hw(p)
        rows.append({"generation": g, "p": p, "q": 1 - p, "AA": AA, "Aa": Aa, "aa": aa})
        p = p * (1 - u) + (1 - p) * v
        p = min(max(p, 0), 1)
    return pd.DataFrame(rows)


def simulate_migration(p0, p_mig, m, generations):
    p = p0
    rows = []
    for g in range(generations + 1):
        AA, Aa, aa = hw(p)
        rows.append({"generation": g, "p": p, "q": 1 - p, "AA": AA, "Aa": Aa, "aa": aa})
        p = (1 - m) * p + m * p_mig
        p = min(max(p, 0), 1)
    return pd.DataFrame(rows)


def simulate_nonrandom(p0, F, generations):
    q = 1 - p0
    H0 = 2 * p0 * q
    rows = []
    for g in range(generations + 1):
        Aa = H0 * (1 - F) ** g
        AA = p0 - Aa / 2
        aa = q - Aa / 2
        rows.append({"generation": g, "p": p0, "q": q, "AA": max(AA, 0), "Aa": max(Aa, 0), "aa": max(aa, 0)})
    return pd.DataFrame(rows)


def simulate_drift(p0, N, generations, runs, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(1, runs + 1):
        p = p0
        for g in range(generations + 1):
            rows.append({"run": r, "generation": g, "p": p, "q": 1 - p})
            if g < generations and 0 < p < 1:
                p = rng.binomial(2 * N, p) / (2 * N)
    run_df = pd.DataFrame(rows)
    summary = run_df.groupby("generation").agg(mean_p=("p", "mean"), min_p=("p", "min"), max_p=("p", "max")).reset_index()
    return run_df, summary


def simulate_topic(topic, params):
    if topic == "① 하디-바인베르크 평형":
        return simulate_basic(params["p0"], params["generations"])
    if topic == "③ 자연선택":
        return simulate_selection(params["p0"], params["s"], params["h"], params["generations"])
    if topic == "④ 돌연변이":
        return simulate_mutation(params["p0"], params["u"], params["v"], params["generations"])
    if topic == "⑤ 유전자 흐름":
        return simulate_migration(params["p0"], params["p_mig"], params["m"], params["generations"])
    if topic == "⑥ 비무작위 교배":
        return simulate_nonrandom(params["p0"], params["F"], params["generations"])
    raise ValueError("지원하지 않는 주제입니다.")


def p_chart(df_a, df_b, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_a["generation"], y=df_a["p"], mode="lines", name="조건 A p", line=dict(color=COL["A"], width=3)))
    fig.add_trace(go.Scatter(x=df_b["generation"], y=df_b["p"], mode="lines", name="조건 B p", line=dict(color=COL["B"], width=3)))
    fig.add_trace(go.Scatter(x=df_a["generation"], y=df_a["q"], mode="lines", name="조건 A q", line=dict(color=COL["A"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=df_b["generation"], y=df_b["q"], mode="lines", name="조건 B q", line=dict(color=COL["B"], width=2, dash="dot")))
    fig.update_layout(title=title, xaxis_title="세대", yaxis_title="대립유전자 빈도", yaxis=dict(range=[0, 1]), height=420, legend=dict(orientation="h", y=1.12))
    return fig


def genotype_chart(df_a, df_b, title):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("조건 A", "조건 B"))
    for c, df in [(1, df_a), (2, df_b)]:
        fig.add_trace(go.Scatter(x=df["generation"], y=df["AA"], mode="lines", name="AA", line=dict(color=COL["AA"], width=2), showlegend=(c == 1)), row=1, col=c)
        fig.add_trace(go.Scatter(x=df["generation"], y=df["Aa"], mode="lines", name="Aa", line=dict(color=COL["Aa"], width=2), showlegend=(c == 1)), row=1, col=c)
        fig.add_trace(go.Scatter(x=df["generation"], y=df["aa"], mode="lines", name="aa", line=dict(color=COL["aa"], width=2), showlegend=(c == 1)), row=1, col=c)
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(title=title, height=420, legend=dict(orientation="h", y=1.12))
    return fig


def drift_chart(run_a, sum_a, run_b, sum_b):
    fig = go.Figure()
    for r, part in run_a.groupby("run"):
        fig.add_trace(go.Scatter(x=part["generation"], y=part["p"], mode="lines", name="A 반복" if r == 1 else "A 반복", line=dict(color=COL["A"], width=1), opacity=.23, showlegend=(r == 1)))
    for r, part in run_b.groupby("run"):
        fig.add_trace(go.Scatter(x=part["generation"], y=part["p"], mode="lines", name="B 반복" if r == 1 else "B 반복", line=dict(color=COL["B"], width=1), opacity=.23, showlegend=(r == 1)))
    fig.add_trace(go.Scatter(x=sum_a["generation"], y=sum_a["mean_p"], mode="lines", name="A 평균", line=dict(color=COL["A"], width=4)))
    fig.add_trace(go.Scatter(x=sum_b["generation"], y=sum_b["mean_p"], mode="lines", name="B 평균", line=dict(color=COL["B"], width=4)))
    fig.update_layout(title="유전적 부동 비교", xaxis_title="세대", yaxis_title="A 대립유전자 빈도 p", yaxis=dict(range=[0, 1]), height=450, legend=dict(orientation="h", y=1.13))
    return fig


def summary_row(label, df, params):
    first, last = df.iloc[0], df.iloc[-1]
    return {
        "조건": label,
        "초기 p": round(float(first["p"]), 4),
        "최종 p": round(float(last["p"]), 4),
        "p 변화량": round(float(last["p"] - first["p"]), 4),
        "최종 q": round(float(last["q"]), 4),
        "최종 AA": round(float(last["AA"]), 4),
        "최종 Aa": round(float(last["Aa"]), 4),
        "최종 aa": round(float(last["aa"]), 4),
        "조건값": ", ".join(f"{k}={v}" for k, v in params.items() if k != "generations"),
    }


def drift_summary(label, run_df, params):
    final = run_df[run_df["generation"] == run_df["generation"].max()]
    return {
        "조건": label,
        "초기 p": round(float(params["p0"]), 4),
        "최종 평균 p": round(float(final["p"].mean()), 4),
        "최종 p 최솟값": round(float(final["p"].min()), 4),
        "최종 p 최댓값": round(float(final["p"].max()), 4),
        "고정 횟수": int((final["p"] >= 1).sum()),
        "소실 횟수": int((final["p"] <= 0).sum()),
        "조건값": ", ".join(f"{k}={v}" for k, v in params.items() if k != "generations"),
    }


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


if "records" not in st.session_state:
    st.session_state.records = []

st.markdown("""
<div class="hero">
<h1>하디-바인베르크 법칙 탐구 시뮬레이터</h1>
<p>평형 조건이 깨질 때 집단의 유전적 구성이 어떻게 변하는지 비교 실험으로 탐구합니다.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("탐구 설정")
    topic = st.selectbox("탐구 주제", TOPICS)
    generations = st.slider("세대 수", 10, 200, 60, 10)
    if topic == "① 하디-바인베르크 평형":
        p0_common = 0.5
    else:
        p0_common = st.slider("공통 초기 p₀(A 빈도)", 0.01, 0.99, 0.50, 0.01)
        st.caption(f"q₀ = {1 - p0_common:.2f}")
    seed = st.number_input("무작위 시드", 1, 99999, 2026, 1)

st.subheader("1. 탐구 질문과 예상 세우기")
st.markdown(f"""
<div class="box">
<b>추천 탐구 질문</b><br>{GUIDE[topic]['q']}<br><br>
<span class="small">그래프를 보기 전에 예상과 이유를 먼저 작성하게 하면 학생 중심 탐구 흐름을 만들 수 있습니다.</span>
</div>
""", unsafe_allow_html=True)

with st.form("plan_form"):
    research_question = st.text_input("나의 탐구 질문", value=GUIDE[topic]["q"])
    prediction = st.text_area("나의 예상", placeholder="예: 집단 크기가 작은 조건에서 p가 더 크게 흔들릴 것이다.")
    prediction_reason = st.text_area("그렇게 예상한 이유", placeholder="예: 작은 집단에서는 우연의 영향이 더 크게 나타날 것 같기 때문이다.")
    st.form_submit_button("탐구 계획 입력 완료")

if not prediction.strip():
    st.warning("학생 활동에서는 그래프를 보기 전에 '나의 예상'을 먼저 작성하게 하는 것을 권장합니다.")

st.subheader("2. 변인 통제 비교 실험 설정")
st.caption("조건 A와 조건 B는 한 가지 변인만 다르게 설정하는 것을 기본으로 합니다.")
col_a, col_b = st.columns(2)
params_a = {"generations": generations}
params_b = {"generations": generations}

if topic == "① 하디-바인베르크 평형":
    with col_a:
        st.markdown("#### 조건 A")
        params_a["p0"] = st.slider("A 조건 p₀", 0.01, 0.99, 0.30, 0.01, key="basic_a")
    with col_b:
        st.markdown("#### 조건 B")
        params_b["p0"] = st.slider("B 조건 p₀", 0.01, 0.99, 0.70, 0.01, key="basic_b")

elif topic == "② 유전적 부동":
    runs = st.slider("반복 실행 횟수", 3, 30, 10, 1)
    with col_a:
        st.markdown("#### 조건 A")
        params_a.update({"p0": p0_common, "N": st.slider("A 조건 집단 크기 N", 5, 1000, 50, 5), "runs": runs})
    with col_b:
        st.markdown("#### 조건 B")
        params_b.update({"p0": p0_common, "N": st.slider("B 조건 집단 크기 N", 5, 1000, 500, 5), "runs": runs})

elif topic == "③ 자연선택":
    h = st.slider("공통 우세 계수 h", 0.0, 1.0, 0.5, 0.05)
    with col_a:
        st.markdown("#### 조건 A")
        params_a.update({"p0": p0_common, "h": h, "s": st.slider("A 조건 선택 계수 s", 0.0, 1.0, 0.10, 0.01)})
    with col_b:
        st.markdown("#### 조건 B")
        params_b.update({"p0": p0_common, "h": h, "s": st.slider("B 조건 선택 계수 s", 0.0, 1.0, 0.40, 0.01)})

elif topic == "④ 돌연변이":
    v = st.slider("공통 역돌연변이율 v(a→A)", 0.0, 0.05, 0.001, 0.0005, format="%.4f")
    with col_a:
        st.markdown("#### 조건 A")
        params_a.update({"p0": p0_common, "v": v, "u": st.slider("A 조건 전향 돌연변이율 u(A→a)", 0.0, 0.05, 0.002, 0.0005, format="%.4f")})
    with col_b:
        st.markdown("#### 조건 B")
        params_b.update({"p0": p0_common, "v": v, "u": st.slider("B 조건 전향 돌연변이율 u(A→a)", 0.0, 0.05, 0.020, 0.0005, format="%.4f")})

elif topic == "⑤ 유전자 흐름":
    p_mig = st.slider("공통 이입 집단 pₘ", 0.01, 0.99, 0.80, 0.01)
    with col_a:
        st.markdown("#### 조건 A")
        params_a.update({"p0": p0_common, "p_mig": p_mig, "m": st.slider("A 조건 이주율 m", 0.0, 0.5, 0.05, 0.01)})
    with col_b:
        st.markdown("#### 조건 B")
        params_b.update({"p0": p0_common, "p_mig": p_mig, "m": st.slider("B 조건 이주율 m", 0.0, 0.5, 0.20, 0.01)})

elif topic == "⑥ 비무작위 교배":
    with col_a:
        st.markdown("#### 조건 A")
        params_a.update({"p0": p0_common, "F": st.slider("A 조건 근친교배 계수 F", 0.0, 1.0, 0.10, 0.05)})
    with col_b:
        st.markdown("#### 조건 B")
        params_b.update({"p0": p0_common, "F": st.slider("B 조건 근친교배 계수 F", 0.0, 1.0, 0.50, 0.05)})

condition_df = pd.DataFrame([{"조건": "A", **params_a}, {"조건": "B", **params_b}])
st.dataframe(condition_df, use_container_width=True, hide_index=True)

st.subheader("3. 시뮬레이션 결과 관찰")

if topic == "② 유전적 부동":
    run_a, sum_a = simulate_drift(params_a["p0"], params_a["N"], generations, params_a["runs"], int(seed))
    run_b, sum_b = simulate_drift(params_b["p0"], params_b["N"], generations, params_b["runs"], int(seed) + 777)
    st.plotly_chart(drift_chart(run_a, sum_a, run_b, sum_b), use_container_width=True)
    summary_df = pd.DataFrame([drift_summary("A", run_a, params_a), drift_summary("B", run_b, params_b)])
else:
    df_a = simulate_topic(topic, params_a)
    df_b = simulate_topic(topic, params_b)
    if topic == "① 하디-바인베르크 평형":
        st.plotly_chart(genotype_chart(df_a, df_b, "평형 상태에서 유전자형 빈도 비교"), use_container_width=True)
    elif topic == "⑥ 비무작위 교배":
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(p_chart(df_a, df_b, "대립유전자 빈도 p, q 변화"), use_container_width=True)
        with c2:
            st.plotly_chart(genotype_chart(df_a, df_b, "유전자형 빈도 변화"), use_container_width=True)
    else:
        st.plotly_chart(p_chart(df_a, df_b, "대립유전자 빈도 p, q 변화"), use_container_width=True)
        with st.expander("유전자형 빈도 그래프도 보기"):
            st.plotly_chart(genotype_chart(df_a, df_b, "유전자형 빈도 변화"), use_container_width=True)
    summary_df = pd.DataFrame([summary_row("A", df_a, params_a), summary_row("B", df_b, params_b)])

st.markdown("#### 결과 요약표")
st.dataframe(summary_df, use_container_width=True, hide_index=True)
st.download_button("결과 요약표 CSV 다운로드", data=csv_bytes(summary_df), file_name="hardy_weinberg_result_summary.csv", mime="text/csv")

st.subheader("4. 관찰 기록과 결론 작성")
with st.form("record_form"):
    changed_variable = st.text_input("이번 비교에서 다르게 한 조건", value=GUIDE[topic]["changed"])
    controlled_variables = st.text_input("같게 유지한 조건", value=GUIDE[topic]["controlled"])
    observation = st.text_area("그래프에서 관찰한 사실", placeholder="예: 조건 A보다 조건 B에서 p가 더 빠르게 증가했다.")
    evidence = st.text_area("그래프나 표에서 찾은 근거", placeholder="예: 60세대 후 조건 A의 p는 0.52, 조건 B의 p는 0.73이었다.")
    interpretation = st.text_area("생물학적 해석 및 결론", placeholder="예: 선택 계수가 클수록 불리한 유전자형이 더 빨리 줄어들기 때문에 대립유전자 빈도 변화가 커진다.")
    save = st.form_submit_button("탐구 기록 저장")

if save:
    st.session_state.records.append({
        "탐구 주제": topic,
        "탐구 질문": research_question,
        "나의 예상": prediction,
        "예상 이유": prediction_reason,
        "다르게 한 조건": changed_variable,
        "같게 유지한 조건": controlled_variables,
        "관찰한 사실": observation,
        "근거": evidence,
        "해석 및 결론": interpretation,
    })
    st.success("탐구 기록이 저장되었습니다.")

if st.session_state.records:
    st.markdown("#### 저장된 탐구 기록")
    record_df = pd.DataFrame(st.session_state.records)
    st.dataframe(record_df, use_container_width=True, hide_index=True)
    st.download_button("탐구 기록 CSV 다운로드", data=csv_bytes(record_df), file_name="hardy_weinberg_inquiry_records.csv", mime="text/csv")
    if st.button("저장된 탐구 기록 모두 삭제"):
        st.session_state.records = []
        st.rerun()

st.subheader("5. 자료 해석형 질문")
for i, question in enumerate(GUIDE[topic]["quiz"], start=1):
    st.text_area(f"질문 {i}. {question}", key=f"question_{topic}_{i}")

with st.expander("해설 보기: 학생 답변 후 열기 권장"):
    st.write(GUIDE[topic]["help"])
    st.markdown("""
- 그래프를 볼 때는 먼저 어떤 변인을 다르게 했는지 확인합니다.
- p, q 같은 대립유전자 빈도와 AA, Aa, aa 같은 유전자형 빈도를 구분합니다.
- 결론에는 그래프나 표의 수치를 근거로 포함합니다.
""")

st.divider()
st.caption("하디-바인베르크 학생 중심 탐구 수업용 Streamlit 웹앱")
