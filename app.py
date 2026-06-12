import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="하디-바인베르크 법칙 탐구",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 커스텀 CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

  /* 사이드바 */
  [data-testid="stSidebar"] {
    background: #0d1b2a;
    color: #e0eaf5;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #c8d8eb !important;
  }

  /* 메인 배경 */
  .main .block-container { background: #f5f7fa; padding-top: 1.5rem; }

  /* 히어로 헤더 */
  .hero-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a3a5c 60%, #0f5a8e 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: "p² + 2pq + q² = 1";
    position: absolute;
    right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    color: rgba(100,200,255,0.18);
    letter-spacing: 0.05em;
    white-space: nowrap;
  }
  .hero-title {
    font-size: 2rem; font-weight: 700;
    color: #ffffff; margin: 0 0 0.4rem;
    letter-spacing: -0.02em;
  }
  .hero-sub {
    font-size: 1rem; font-weight: 300;
    color: #a8c8e8; margin: 0;
  }

  /* 섹션 카드 */
  .card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-left: 4px solid #0f5a8e;
  }
  .card-warn { border-left-color: #e06c1e; }
  .card-ok   { border-left-color: #1e9e5a; }

  /* 수식 블록 */
  .formula-box {
    background: #eaf2fb;
    border: 1px solid #b0cfe8;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    color: #0d2a45;
    margin: 0.6rem 0 1rem;
  }

  /* 배지 */
  .badge {
    display: inline-block;
    background: #0f5a8e22;
    color: #0f5a8e;
    border: 1px solid #0f5a8e55;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.15rem 0.75rem;
    margin-right: 0.4rem;
    letter-spacing: 0.03em;
  }
  .badge-warn { background:#e06c1e22; color:#c05010; border-color:#e06c1e55; }

  /* 지표 숫자 */
  .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.8rem 0; }
  .metric-box {
    background: #f0f5fb;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    text-align: center;
    min-width: 100px;
  }
  .metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem; font-weight: 600; color: #0d2a45;
  }
  .metric-lbl { font-size: 0.75rem; color: #6e8aaa; margin-top: 0.1rem; }

  /* 탭 */
  .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.4rem 1.1rem;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────
def hw_genotype_freq(p):
    q = 1 - p
    return p**2, 2*p*q, q**2   # AA, Aa, aa

def simulate_drift(N, p0, generations, n_runs=10):
    """유전적 부동 시뮬레이션 (이항분포 샘플링)"""
    results = []
    for _ in range(n_runs):
        p = p0
        traj = [p]
        for _ in range(generations):
            if p <= 0 or p >= 1:
                traj.append(p)
                continue
            allele_count = np.random.binomial(2 * N, p)
            p = allele_count / (2 * N)
            traj.append(p)
        results.append(traj)
    return results

def simulate_selection(p0, s, h, generations):
    """자연선택 시뮬레이션
    fitness: AA=1, Aa=1-hs, aa=1-s
    """
    p = p0
    traj = [p]
    for _ in range(generations):
        q = 1 - p
        w_AA = 1.0
        w_Aa = 1.0 - h * s
        w_aa = 1.0 - s
        w_bar = p**2 * w_AA + 2*p*q * w_Aa + q**2 * w_aa
        if w_bar == 0:
            break
        p_new = (p**2 * w_AA + p*q * w_Aa) / w_bar
        p = p_new
        traj.append(p)
    return traj

def simulate_mutation(p0, u, v, generations):
    """돌연변이 시뮬레이션 (A→a: u, a→A: v)"""
    p = p0
    traj = [p]
    for _ in range(generations):
        p = p * (1 - u) + (1 - p) * v
        traj.append(p)
    return traj

def simulate_migration(p0, p_m, m, generations):
    """유전자 흐름(이주) 시뮬레이션"""
    p = p0
    traj = [p]
    for _ in range(generations):
        p = (1 - m) * p + m * p_m
        traj.append(p)
    return traj

def simulate_nonrandom_mating(p0, F_inbreeding, generations):
    """비무작위 교배 (근친교배 계수 F) — 세대별 누적 모델.

    매 세대 이형접합체의 비율이 (1 - F)배씩 감소한다.
    - H_t = H_0 * (1 - F)^t  (H: 이형접합체 빈도)
    - 대립유전자 빈도 p는 변하지 않음
    - 남은 빈도(1 - H_t)는 p:q 비율로 AA와 aa에 분배
    """
    p = p0
    q = 1 - p
    H0 = 2 * p * q          # 초기 HW 이형접합체 빈도

    traj_p  = [p]
    traj_AA = [p**2]         # 0세대: HW 기대값
    traj_Aa = [H0]
    traj_aa = [q**2]

    for t in range(1, generations + 1):
        # 이형접합체 빈도가 매 세대 (1-F) 배 감소
        Aa = H0 * (1 - F_inbreeding) ** t
        # 나머지 (1 - Aa)를 p:q 비율로 동형접합체에 분배
        AA = p - Aa / 2          # p = AA + Aa/2 항상 성립
        aa = q - Aa / 2          # q = aa + Aa/2 항상 성립
        traj_AA.append(max(AA, 0))
        traj_Aa.append(max(Aa, 0))
        traj_aa.append(max(aa, 0))
        traj_p.append(p)         # 대립유전자 빈도는 불변

    return traj_p, traj_AA, traj_Aa, traj_aa

COLORS = {
    "AA": "#0f5a8e",
    "Aa": "#1e9e5a",
    "aa": "#e06c1e",
    "p":  "#0f5a8e",
    "q":  "#c0392b",
    "hw": "#888888",
}

GEN_RANGE = list(range(0, 101))


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 하디-바인베르크 탐구")
    st.markdown("---")
    topic = st.selectbox(
        "📚 주제 선택",
        [
            "① 하디-바인베르크 평형 (기본)",
            "② 유전적 부동 (소집단)",
            "③ 자연선택",
            "④ 돌연변이",
            "⑤ 유전자 흐름 (이주)",
            "⑥ 비무작위 교배 (근친교배)",
        ],
    )
    st.markdown("---")
    st.markdown("**초기 대립유전자 빈도**")
    p0 = st.slider("p₀ (A 대립유전자 빈도)", 0.01, 0.99, 0.5, 0.01,
                   help="초기 우성 대립유전자(A) 빈도. q₀ = 1 – p₀")
    q0 = round(1 - p0, 4)
    st.markdown(f"<div style='color:#c8d8eb;font-size:0.9rem'>q₀ = {q0}</div>",
                unsafe_allow_html=True)

    generations = st.slider("세대 수", 10, 200, 50, 10)

    # 주제별 파라미터
    st.markdown("---")
    st.markdown("**시나리오 파라미터**")

    if "② 유전적 부동" in topic:
        N = st.slider("집단 크기 (N)", 5, 1000, 50, 5)
        n_runs = st.slider("시뮬레이션 반복 횟수", 3, 20, 8)

    elif "③ 자연선택" in topic:
        s = st.slider("선택 계수 (s)", 0.0, 1.0, 0.1, 0.01,
                      help="aa 유전자형의 적합도 감소량")
        h = st.slider("우세 계수 (h)", 0.0, 1.0, 0.5, 0.01,
                      help="0=완전우성, 0.5=공우성, 1=완전열성")
        mode_label = {0.0:"완전 우성", 0.5:"공우성 (불완전)", 1.0:"완전 열성"}.get(h, "중간 우성")
        st.caption(f"현재 모드: **{mode_label}**")

    elif "④ 돌연변이" in topic:
        u = st.slider("돌연변이율 A→a (u)", 0.0, 0.05, 0.001, 0.0001, format="%.4f")
        v = st.slider("역돌연변이율 a→A (v)", 0.0, 0.05, 0.0001, 0.0001, format="%.4f")

    elif "⑤ 유전자 흐름" in topic:
        p_mig = st.slider("이입 집단의 A 빈도 (p_m)", 0.01, 0.99, 0.8, 0.01)
        m = st.slider("이주율 (m)", 0.0, 0.5, 0.05, 0.01)

    elif "⑥ 비무작위 교배" in topic:
        F = st.slider("근친교배 계수 (F)", 0.0, 1.0, 0.3, 0.05)

    st.markdown("---")
    st.caption("고등학교 생물 수업용 시뮬레이터\n\n하디-바인베르크 법칙 탐구")


# ── 메인 ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">하디-바인베르크 법칙 탐구 시뮬레이터</div>
  <div class="hero-sub">전제 조건이 깨질 때 집단 유전자 빈도는 어떻게 변할까?</div>
</div>
""", unsafe_allow_html=True)

tab_sim, tab_theory, tab_quiz = st.tabs(["📊 시뮬레이션", "📖 이론 정리", "🧪 퀴즈"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – 시뮬레이션
# ════════════════════════════════════════════════════════════════════════════════
with tab_sim:

    # ── ① 평형 (기본) ──────────────────────────────────────────────────────────
    if "① 하디-바인베르크" in topic:
        st.markdown('<div class="card card-ok">', unsafe_allow_html=True)
        st.markdown("### ① 하디-바인베르크 평형 상태")
        st.markdown(
            "모든 전제 조건이 충족된 이상적 집단에서는 "
            "세대가 지나도 **대립유전자 빈도와 유전자형 빈도가 일정**합니다."
        )
        st.markdown(
            '<div class="formula-box">p² (AA) + 2pq (Aa) + q² (aa) = 1</div>',
            unsafe_allow_html=True,
        )

        AA, Aa, aa = hw_genotype_freq(p0)
        q0_ = 1 - p0

        st.markdown('<div class="metric-row">', unsafe_allow_html=True)
        cols = st.columns(5)
        cols[0].metric("p (A 빈도)", f"{p0:.3f}")
        cols[1].metric("q (a 빈도)", f"{q0_:.3f}")
        cols[2].metric("AA 빈도 (p²)", f"{AA:.3f}")
        cols[3].metric("Aa 빈도 (2pq)", f"{Aa:.3f}")
        cols[4].metric("aa 빈도 (q²)", f"{aa:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

        # 파이 차트 (유전자형 빈도)
        fig_pie = go.Figure(go.Pie(
            labels=["AA (p²)", "Aa (2pq)", "aa (q²)"],
            values=[AA, Aa, aa],
            marker_colors=[COLORS["AA"], COLORS["Aa"], COLORS["aa"]],
            hole=0.38,
            textinfo="label+percent",
            textfont_size=14,
        ))
        fig_pie.update_layout(
            title="유전자형 빈도 분포",
            height=380,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # 세대별 평형 유지 시각화 (항상 수평)
        gens = list(range(generations + 1))
        fig_line = go.Figure()
        for label, val, col in [("AA (p²)", AA, COLORS["AA"]),
                                 ("Aa (2pq)", Aa, COLORS["Aa"]),
                                 ("aa (q²)", aa, COLORS["aa"])]:
            fig_line.add_trace(go.Scatter(
                x=gens, y=[val] * (generations + 1),
                mode="lines", name=label,
                line=dict(color=col, width=2.5),
            ))
        fig_line.update_layout(
            title="세대별 유전자형 빈도 (평형 유지)",
            xaxis_title="세대", yaxis_title="빈도",
            yaxis=dict(range=[0, 1]),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f9fbfd",
            legend=dict(orientation="h", y=1.1),
        )

        c1, c2 = st.columns([1, 1.4])
        c1.plotly_chart(fig_pie, use_container_width=True)
        c2.plotly_chart(fig_line, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── ② 유전적 부동 ──────────────────────────────────────────────────────────
    elif "② 유전적 부동" in topic:
        st.markdown('<div class="card card-warn">', unsafe_allow_html=True)
        st.markdown("### ② 유전적 부동 — 소집단 효과")
        st.markdown(
            "집단이 **작을수록** 우연한 샘플링 오차로 대립유전자 빈도가 크게 흔들리며, "
            "결국 **고정(fixation)** 또는 **소실(loss)** 이 일어납니다."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        trajs = simulate_drift(N, p0, generations, n_runs)

        fig = go.Figure()
        fixed = sum(1 for t in trajs if t[-1] >= 1.0)
        lost  = sum(1 for t in trajs if t[-1] <= 0.0)

        for i, traj in enumerate(trajs):
            color = "#0f5a8e" if traj[-1] >= 1.0 else ("#e06c1e" if traj[-1] <= 0.0 else "#aaaaaa")
            fig.add_trace(go.Scatter(
                x=list(range(len(traj))), y=traj,
                mode="lines",
                line=dict(color=color, width=1.5),
                opacity=0.7,
                name=f"실행 {i+1}",
                showlegend=False,
            ))
        # HW 평형선
        fig.add_hline(y=p0, line_dash="dot", line_color="#555555",
                      annotation_text=f"HW 평형 p={p0}", annotation_position="right")

        fig.update_layout(
            title=f"유전적 부동 시뮬레이션 (N={N}, {n_runs}회 반복)",
            xaxis_title="세대", yaxis_title="A 대립유전자 빈도 (p)",
            yaxis=dict(range=[-0.05, 1.05]),
            height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fbfd",
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("A 고정 (p→1)", f"{fixed}/{n_runs}회", delta=None)
        c2.metric("A 소실 (p→0)", f"{lost}/{n_runs}회")
        c3.metric("집단 크기", f"N = {N}")

        st.info(
            f"💡 **집단 크기가 작을수록** 부동 효과가 크며, "
            f"N={N}일 때 대립유전자 1개의 빈도 변화 최소 단위는 **{1/(2*N):.4f}** 입니다."
        )

    # ── ③ 자연선택 ─────────────────────────────────────────────────────────────
    elif "③ 자연선택" in topic:
        st.markdown('<div class="card card-warn">', unsafe_allow_html=True)
        st.markdown("### ③ 자연선택")
        st.markdown(
            "적합도(fitness)가 유전자형마다 다를 때, "
            "**불리한 대립유전자**는 점차 줄어들고 **유리한 대립유전자**는 증가합니다."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        traj = simulate_selection(p0, s, h, generations)
        gens = list(range(len(traj)))

        fig = make_subplots(rows=1, cols=2, subplot_titles=("A 대립유전자 빈도 변화", "유전자형 빈도 변화"))

        fig.add_trace(go.Scatter(x=gens, y=traj, mode="lines",
                                 line=dict(color=COLORS["p"], width=2.5), name="p (A)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=gens, y=[1-v for v in traj], mode="lines",
                                 line=dict(color=COLORS["q"], width=2.5), name="q (a)"), row=1, col=1)
        fig.add_hline(y=p0, line_dash="dot", line_color="#888888",
                      annotation_text="초기값", row=1, col=1)

        AA_t = [v**2 for v in traj]
        Aa_t = [2*v*(1-v) for v in traj]
        aa_t = [(1-v)**2 for v in traj]

        fig.add_trace(go.Scatter(x=gens, y=AA_t, mode="lines",
                                 line=dict(color=COLORS["AA"], width=2), name="AA"), row=1, col=2)
        fig.add_trace(go.Scatter(x=gens, y=Aa_t, mode="lines",
                                 line=dict(color=COLORS["Aa"], width=2), name="Aa"), row=1, col=2)
        fig.add_trace(go.Scatter(x=gens, y=aa_t, mode="lines",
                                 line=dict(color=COLORS["aa"], width=2), name="aa"), row=1, col=2)

        fig.update_yaxes(range=[0, 1])
        fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fbfd",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        # 적합도 테이블
        fitness_df = pd.DataFrame({
            "유전자형": ["AA", "Aa", "aa"],
            "적합도": [1.0, round(1 - h*s, 4), round(1 - s, 4)],
            "설명": ["기준 (가장 유리)", f"1 − hs = {1-h*s:.3f}", f"1 − s = {1-s:.3f}"],
        })
        st.dataframe(fitness_df, use_container_width=True, hide_index=True)

        st.info(
            f"💡 s={s}, h={h}일 때 aa의 적합도는 **{1-s:.3f}** 입니다. "
            f"선택이 강할수록(s↑), 열성 대립유전자는 빠르게 감소합니다."
        )

    # ── ④ 돌연변이 ─────────────────────────────────────────────────────────────
    elif "④ 돌연변이" in topic:
        st.markdown('<div class="card card-warn">', unsafe_allow_html=True)
        st.markdown("### ④ 돌연변이")
        st.markdown(
            "돌연변이는 대립유전자 빈도를 변화시키는 원동력입니다. "
            "전향 돌연변이(A→a)와 역돌연변이(a→A)가 균형을 이루는 **돌연변이 평형**에 도달합니다."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        traj = simulate_mutation(p0, u, v, generations)
        # 이론적 평형값
        p_eq = v / (u + v) if (u + v) > 0 else p0
        gens = list(range(len(traj)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gens, y=traj, mode="lines",
                                 line=dict(color=COLORS["p"], width=2.5), name="p (A 빈도)"))
        fig.add_hline(y=p_eq, line_dash="dash", line_color="#1e9e5a",
                      annotation_text=f"돌연변이 평형 p* = {p_eq:.4f}",
                      annotation_position="right", annotation_font_color="#1e9e5a")
        fig.add_hline(y=p0, line_dash="dot", line_color="#888888",
                      annotation_text=f"초기값 p₀ = {p0}")
        fig.update_layout(
            title="돌연변이에 의한 대립유전자 빈도 변화",
            xaxis_title="세대", yaxis_title="A 빈도 (p)",
            yaxis=dict(range=[0, 1]), height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fbfd",
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("전향 돌연변이율 u (A→a)", f"{u:.4f}")
        c2.metric("역돌연변이율 v (a→A)", f"{v:.4f}")
        c3.metric("이론적 평형 p*", f"{p_eq:.4f}")

        st.info("💡 돌연변이율은 매우 낮아 단독으로는 빈도 변화가 느리지만, **진화의 원재료(raw material)**를 제공합니다.")

    # ── ⑤ 유전자 흐름 ─────────────────────────────────────────────────────────
    elif "⑤ 유전자 흐름" in topic:
        st.markdown('<div class="card card-warn">', unsafe_allow_html=True)
        st.markdown("### ⑤ 유전자 흐름 (이주)")
        st.markdown(
            "다른 집단에서 이주가 일어나면, 수용 집단의 대립유전자 빈도가 "
            "**이입 집단의 빈도 쪽으로 수렴**합니다."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        traj = simulate_migration(p0, p_mig, m, generations)
        gens = list(range(len(traj)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gens, y=traj, mode="lines",
                                 line=dict(color=COLORS["p"], width=2.5), name="수용 집단 p"))
        fig.add_hline(y=p_mig, line_dash="dash", line_color="#1e9e5a",
                      annotation_text=f"이입 집단 p_m = {p_mig}",
                      annotation_position="right", annotation_font_color="#1e9e5a")
        fig.add_hline(y=p0, line_dash="dot", line_color="#888888",
                      annotation_text=f"초기값 p₀ = {p0}")
        fig.update_layout(
            title=f"유전자 흐름 시뮬레이션 (이주율 m = {m})",
            xaxis_title="세대", yaxis_title="A 빈도 (p)",
            yaxis=dict(range=[0, 1]), height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fbfd",
        )
        st.plotly_chart(fig, use_container_width=True)

        p_final = traj[-1]
        st.metric("최종 세대 p", f"{p_final:.4f}", delta=f"{p_final-p0:+.4f} vs 초기값")
        st.info(f"💡 이주율이 높을수록(m↑) 두 집단의 빈도는 빠르게 동질화됩니다. 지리적 격리는 이를 막아 종 분화를 촉진합니다.")

    # ── ⑥ 비무작위 교배 ────────────────────────────────────────────────────────
    elif "⑥ 비무작위 교배" in topic:
        st.markdown('<div class="card card-warn">', unsafe_allow_html=True)
        st.markdown("### ⑥ 비무작위 교배 (근친교배)")
        st.markdown(
            "무작위 교배가 이루어지지 않으면 (근친교배 계수 F > 0), "
            "**대립유전자 빈도는 변하지 않지만** 유전자형 빈도가 변합니다. "
            "동형접합체(AA, aa)가 증가하고 이형접합체(Aa)가 감소합니다."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        traj_p, traj_AA, traj_Aa, traj_aa = simulate_nonrandom_mating(p0, F, generations)
        # HW 기대값 (F=0일 때 평형값)
        AA_hw, Aa_hw, aa_hw = hw_genotype_freq(p0)
        q0_ = 1 - p0
        gens = list(range(generations + 1))   # 0세대부터 시작

        # ── 두 그래프를 나란히 배치 ──────────────────────────────────────────
        left, right = st.columns(2)

        # 왼쪽: 유전자형 빈도 변화
        with left:
            fig_geno = go.Figure()
            fig_geno.add_trace(go.Scatter(
                x=gens, y=traj_AA, mode="lines",
                line=dict(color=COLORS["AA"], width=2.5), name="AA"))
            fig_geno.add_trace(go.Scatter(
                x=gens, y=traj_Aa, mode="lines",
                line=dict(color=COLORS["Aa"], width=2.5), name="Aa"))
            fig_geno.add_trace(go.Scatter(
                x=gens, y=traj_aa, mode="lines",
                line=dict(color=COLORS["aa"], width=2.5), name="aa"))

            # HW 기대값 점선 (F=0 기준선)
            for label, val, col in [("AA (F=0)", AA_hw, COLORS["AA"]),
                                     ("Aa (F=0)", Aa_hw, COLORS["Aa"]),
                                     ("aa (F=0)", aa_hw, COLORS["aa"])]:
                fig_geno.add_hline(y=val, line_dash="dot", line_color=col, opacity=0.35,
                                   annotation_text=label, annotation_font_size=10)

            fig_geno.update_layout(
                title="📉 유전자형 빈도 변화",
                xaxis_title="세대", yaxis_title="유전자형 빈도",
                yaxis=dict(range=[0, 1]), height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fbfd",
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_geno, use_container_width=True)

        # 오른쪽: 대립유전자 빈도 — 수평 유지
        with right:
            # 각 세대에서 p와 q를 AA·Aa·aa로부터 역산 (= 항상 p0, q0)
            p_from_geno = [AA + Aa/2 for AA, Aa in zip(traj_AA, traj_Aa)]
            q_from_geno = [aa + Aa/2 for aa, Aa in zip(traj_aa, traj_Aa)]

            fig_allele = go.Figure()
            fig_allele.add_trace(go.Scatter(
                x=gens, y=p_from_geno, mode="lines",
                line=dict(color=COLORS["p"], width=3), name=f"p (A 빈도)"))
            fig_allele.add_trace(go.Scatter(
                x=gens, y=q_from_geno, mode="lines",
                line=dict(color=COLORS["q"], width=3), name=f"q (a 빈도)"))

            # 기준선 강조
            fig_allele.add_hline(
                y=p0, line_dash="dot", line_color=COLORS["p"], opacity=0.3,
                annotation_text=f"p₀ = {p0}", annotation_font_size=11)
            fig_allele.add_hline(
                y=q0_, line_dash="dot", line_color=COLORS["q"], opacity=0.3,
                annotation_text=f"q₀ = {q0_:.2f}", annotation_font_size=11)

            # 불변 강조 박스 텍스트
            fig_allele.add_annotation(
                x=generations * 0.5, y=0.5,
                text="p, q 불변 ✔",
                showarrow=False,
                font=dict(size=18, color="#1e9e5a"),
                opacity=0.18,
            )

            fig_allele.update_layout(
                title="📌 대립유전자 빈도 — 세대 내내 일정",
                xaxis_title="세대", yaxis_title="대립유전자 빈도",
                yaxis=dict(range=[0, 1]), height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fbfd",
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_allele, use_container_width=True)

        # ── 수식 및 지표 ─────────────────────────────────────────────────────
        H0 = 2 * p0 * q0_
        Aa_final = traj_Aa[-1]
        st.markdown(
            f'<div class="formula-box">'
            f'H<sub>t</sub> = H<sub>0</sub> × (1 − F)<sup>t</sup> &nbsp;=&nbsp; '
            f'{H0:.3f} × (1 − {F})<sup>{generations}</sup> &nbsp;=&nbsp; <strong>{Aa_final:.4f}</strong>'
            f'&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;'
            f'p = AA + Aa/2 = <strong>{p_from_geno[-1]:.4f}</strong> (불변)'
            f'</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("p (A 빈도)", f"{p_from_geno[-1]:.4f}", delta="0.0000 변화없음")
        col2.metric("q (a 빈도)", f"{q_from_geno[-1]:.4f}", delta="0.0000 변화없음")
        col3.metric("AA vs HW", f"{traj_AA[-1]:.3f}", delta=f"{traj_AA[-1]-AA_hw:+.3f}")
        col4.metric("Aa vs HW", f"{traj_Aa[-1]:.3f}", delta=f"{traj_Aa[-1]-Aa_hw:+.3f}")
        col5.metric("aa vs HW", f"{traj_aa[-1]:.3f}", delta=f"{traj_aa[-1]-aa_hw:+.3f}")

        st.info(
            f"💡 오른쪽 그래프를 보세요. p와 q는 F={F}, {generations}세대 후에도 "
            f"**p={p0}, q={q0_:.2f}로 완전히 일정**합니다. "
            "근친교배는 유전자형 빈도(왼쪽)만 바꿀 뿐, 대립유전자 빈도 자체는 변화시키지 않습니다."
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – 이론 정리
# ════════════════════════════════════════════════════════════════════════════════
with tab_theory:
    st.markdown("## 하디-바인베르크 법칙 — 핵심 정리")

    st.markdown('<div class="card card-ok">', unsafe_allow_html=True)
    st.markdown("### 🔑 핵심 공식")
    st.markdown(
        '<div class="formula-box">'
        'p + q = 1 &nbsp;&nbsp;&nbsp; (대립유전자 빈도)<br>'
        'p² + 2pq + q² = 1 &nbsp;&nbsp;&nbsp; (유전자형 빈도)'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📋 5가지 전제 조건과 위반 효과")
    conditions = [
        ("무한 집단 크기", "유전적 부동", "소집단에서는 우연한 빈도 변화 → 고정 또는 소실"),
        ("자연선택 없음", "자연선택", "유전자형별 적합도 차이 → 유리한 대립유전자 증가"),
        ("돌연변이 없음", "돌연변이", "새 대립유전자 생성 → 빈도 서서히 변화"),
        ("이주 없음", "유전자 흐름", "외부 유전자 유입 → 집단 빈도 동질화"),
        ("무작위 교배", "비무작위 교배", "근친교배 시 동형접합체 증가, 이형접합체 감소"),
    ]
    for i, (cond, violation, effect) in enumerate(conditions, 1):
        st.markdown(
            f'<div class="card" style="margin-bottom:0.7rem">'
            f'<span class="badge">전제 {i}</span>'
            f'<span class="badge badge-warn">위반: {violation}</span><br>'
            f'<strong>전제:</strong> {cond}<br>'
            f'<span style="color:#555;font-size:0.92rem">▶ {effect}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # pq 삼각형 시각화
    st.markdown("### 📐 p-q 관계와 유전자형 빈도")
    p_vals = np.linspace(0, 1, 200)
    q_vals = 1 - p_vals
    AA_vals = p_vals**2
    Aa_vals = 2*p_vals*q_vals
    aa_vals = q_vals**2

    fig_theory = go.Figure()
    fig_theory.add_trace(go.Scatter(x=p_vals, y=AA_vals, name="AA (p²)",
                                    line=dict(color=COLORS["AA"], width=2.5)))
    fig_theory.add_trace(go.Scatter(x=p_vals, y=Aa_vals, name="Aa (2pq)",
                                    line=dict(color=COLORS["Aa"], width=2.5)))
    fig_theory.add_trace(go.Scatter(x=p_vals, y=aa_vals, name="aa (q²)",
                                    line=dict(color=COLORS["aa"], width=2.5)))
    fig_theory.add_vline(x=p0, line_dash="dash", line_color="#333333",
                         annotation_text=f"현재 p₀={p0}")
    fig_theory.update_layout(
        xaxis_title="A 대립유전자 빈도 (p)",
        yaxis_title="유전자형 빈도",
        yaxis=dict(range=[0, 1]),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fbfd",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_theory, use_container_width=True)
    st.caption("Aa(2pq)는 p=0.5일 때 최대(0.5)이며, AA와 aa는 각각 p→1, p→0일 때 최대가 됩니다.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 – 퀴즈
# ════════════════════════════════════════════════════════════════════════════════
with tab_quiz:
    st.markdown("## 🧪 개념 확인 퀴즈")
    st.markdown("각 문제를 풀고 정답을 확인하세요.")

    quizzes = [
        {
            "q": "Q1. 하디-바인베르크 평형이 유지되기 위한 전제 조건이 **아닌** 것은?",
            "options": ["무한 집단 크기", "자연선택 없음", "높은 돌연변이율", "무작위 교배"],
            "answer": 2,
            "explanation": "하디-바인베르크 평형은 **돌연변이가 없음**을 전제합니다. 높은 돌연변이율은 오히려 평형을 깨는 요인입니다.",
        },
        {
            "q": "Q2. 집단에서 A(우성) 대립유전자 빈도 p=0.6일 때, HW 평형에서 이형접합체(Aa)의 빈도는?",
            "options": ["0.36", "0.48", "0.16", "0.60"],
            "answer": 1,
            "explanation": "Aa = 2pq = 2 × 0.6 × 0.4 = **0.48**",
        },
        {
            "q": "Q3. 소집단에서 우연에 의해 대립유전자 빈도가 변하는 현상은?",
            "options": ["자연선택", "유전자 흐름", "유전적 부동", "근친교배"],
            "answer": 2,
            "explanation": "**유전적 부동(genetic drift)**은 집단 크기가 작을수록 크게 나타납니다.",
        },
        {
            "q": "Q4. 근친교배가 지속될 때 나타나는 변화로 옳은 것은?",
            "options": [
                "A 대립유전자 빈도가 증가한다",
                "이형접합체(Aa) 빈도가 감소한다",
                "자연선택이 강해진다",
                "집단이 무한히 커진다",
            ],
            "answer": 1,
            "explanation": "근친교배는 대립유전자 **빈도는 변화시키지 않지만**, 이형접합체를 줄이고 동형접합체를 늘립니다.",
        },
        {
            "q": "Q5. 두 집단 사이에 이주가 활발할 때 예상되는 결과는?",
            "options": [
                "두 집단의 대립유전자 빈도가 달라진다",
                "두 집단의 대립유전자 빈도가 비슷해진다",
                "돌연변이가 증가한다",
                "자연선택이 약해진다",
            ],
            "answer": 1,
            "explanation": "유전자 흐름(이주)은 집단 간 유전적 차이를 줄여 **빈도를 동질화**합니다.",
        },
    ]

    for i, qz in enumerate(quizzes):
        with st.expander(qz["q"], expanded=False):
            choice = st.radio(
                "정답 선택", qz["options"],
                key=f"quiz_{i}",
                index=None,
            )
            if choice is not None:
                if qz["options"].index(choice) == qz["answer"]:
                    st.success(f"✅ 정답! — {qz['explanation']}")
                else:
                    st.error(f"❌ 오답. {qz['explanation']}")
