"""
5G SON Dashboard — Final Clean Version
Run: python -m streamlit run 5g_son_dashboard.py
"""
import streamlit as st
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
import pandas as pd

st.set_page_config(page_title="5G SON", page_icon="📡", layout="wide")
st.markdown("""
<style>
/* Compact layout */
.block-container{padding-top:.3rem!important;padding-bottom:1rem!important}
section[data-testid="stSidebar"]{width:260px!important;min-width:260px!important}
section[data-testid="stSidebar"] .block-container{padding-top:.5rem!important;padding-left:.8rem!important;padding-right:.8rem!important}
/* Smaller sidebar text */
section[data-testid="stSidebar"] h3{font-size:13px!important;margin:4px 0 2px 0!important}
section[data-testid="stSidebar"] p{font-size:11px!important;margin:2px 0!important}
section[data-testid="stSidebar"] .stButton button{padding:4px 8px!important;font-size:12px!important}
section[data-testid="stSidebar"] .stSelectbox{margin-bottom:2px!important}
section[data-testid="stSidebar"] hr{margin:6px 0!important}
/* Log cards */
.log-ok  {background:#071a0e;border-left:3px solid #27ae60;border-radius:6px;padding:5px 9px;margin:2px 0;font-size:11.5px;color:#a9dfbf}
.log-ok b{color:#2ecc71}
.log-bad {background:#1a0707;border-left:3px solid #e74c3c;border-radius:6px;padding:5px 9px;margin:2px 0;font-size:11.5px;color:#f1948a}
.log-bad b{color:#ff6b6b}
.log-info{background:#071020;border-left:3px solid #3498db;border-radius:6px;padding:5px 9px;margin:2px 0;font-size:11.5px;color:#85c1e9}
.log-info b{color:#5dade2}
.scr{max-height:380px;overflow-y:auto}
.atk{background:#2a0000;border:2px solid #e74c3c;border-radius:7px;padding:6px 12px;color:#ff8888;font-weight:600;text-align:center;margin:4px 0;font-size:12px}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────
EDGES = [
    (1,2,100),(1,3,80),(2,4,90),(2,5,70),(3,5,60),(3,6,110),
    (4,7,85),(5,7,95),(5,8,75),(6,8,100),(6,9,80),(7,10,90),
    (8,10,70),(8,11,85),(9,11,60),(10,12,100),(11,12,90),
    (4,6,65),(3,9,55),(2,8,40),
]
POS = {
    1:(.12,.90),2:(.40,.95),3:(.12,.65),4:(.68,.90),
    5:(.40,.70),6:(.12,.40),7:(.68,.65),8:(.40,.42),
    9:(.12,.15),10:(.68,.40),11:(.40,.15),12:(.68,.15),
}
LABELS = {n:f"gNB{n}" for n in range(1,13)}
CAP    = {(u,v):c for u,v,c in EDGES}   # fixed forever
THRESH = 0.80

# ──────────────────────────────────────────────────────
#  PURE HELPER FUNCTIONS
#  Every function takes T dict, returns NEW T dict.
#  T values are ALWAYS integers between 5 and capacity.
# ──────────────────────────────────────────────────────

def make_traffic(seed=42):
    """Fresh traffic at 30–65% of capacity. Returns T and BASE (frozen copy)."""
    rng = random.Random(seed)
    T = {}
    for u,v,c in EDGES:
        T[(u,v)] = rng.randint(int(c*0.30), int(c*0.65))
    return T, dict(T)


def safe(T):
    """Clamp every value to [5 … capacity]. Call this after EVERY change."""
    return {k: max(5, min(CAP[k], v)) for k, v in T.items()}


def congested(T):
    return [(u,v) for (u,v) in CAP if T[(u,v)] / CAP[(u,v)] >= THRESH]


def do_attack(T, node):
    """Set all links of `node` to exactly capacity (100% = congested)."""
    T2 = dict(T)
    for u,v,c in EDGES:
        if u == node or v == node:
            T2[(u,v)] = CAP[(u,v)]   # 100% — always >= THRESH(80%)
    return safe(T2)


def do_heal(BASE):
    """Reset ALL links to frozen BASE values — guaranteed all green."""
    return safe(dict(BASE))


def do_heal_node(T, BASE, node):
    """Heal only the links connected to a specific node — reset them to base."""
    T2 = dict(T)
    for u,v,c in EDGES:
        if u == node or v == node:
            T2[(u,v)] = BASE[(u,v)]   # restore this link to base value
    return safe(T2)


def do_congest(T, u, v):
    """Force one link to 100% capacity → red."""
    T2 = dict(T)
    T2[(u,v)] = CAP[(u,v)]
    return safe(T2)


def do_clear(T, u, v):
    """Force one link to 35% capacity → green."""
    T2 = dict(T)
    T2[(u,v)] = max(5, int(CAP[(u,v)] * 0.35))
    return safe(T2)


def do_reroute(T):
    """
    For each congested link:
      1. Remove it from graph
      2. Find alternate path via Dijkstra (weighted by load)
      3. Move overflow to alternate path
      4. safe() clamp immediately — no accumulation possible
    Returns (new_T, rerouted_list, log_list)
    """
    T2       = dict(T)
    rerouted = []
    log      = []

    cgs = congested(T2)
    if not cgs:
        log.append(("ok", "✅ Network already healthy — no congestion found"))
        return safe(T2), rerouted, log

    for (u, v) in cgs:
        overflow = T2[(u,v)] - CAP[(u,v)]
        if overflow <= 0:
            continue

        # graph without congested edge
        G = nx.Graph()
        G.add_nodes_from(range(1, 13))
        for a, b, _ in EDGES:
            if (a,b) == (u,v):
                continue
            w = T2[(a,b)] / CAP[(a,b)]
            G.add_edge(a, b, weight=round(w, 4))

        try:
            path = nx.dijkstra_path(G, u, v, weight='weight')
        except Exception:
            log.append(("bad", f"⚠️ No path for {u}↔{v}"))
            continue

        hops  = max(1, len(path) - 1)
        share = max(1, overflow // hops)

        for i in range(hops):
            a, b = path[i], path[i+1]
            k    = (a,b) if (a,b) in CAP else (b,a)
            if k in CAP:
                T2[k] = T2[k] + share

        T2[(u,v)] = max(5, T2[(u,v)] - overflow)

        # CLAMP AFTER EVERY SINGLE LINK — no snowballing
        T2 = safe(T2)

        cleared = T2[(u,v)] / CAP[(u,v)] < THRESH
        rerouted.append((u,v))
        log.append((
            "ok" if cleared else "bad",
            f"{'✅' if cleared else '⚠️'} {u}↔{v} → {'→'.join(map(str,path))} (moved {share})"
        ))

    return safe(T2), rerouted, log


# ──────────────────────────────────────────────────────
#  DRAWING
# ──────────────────────────────────────────────────────

def draw(T, title="", attacked=None, rerouted=None):
    rset = set()
    if rerouted:
        for u,v in rerouted:
            rset.add((u,v)); rset.add((v,u))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    G = nx.Graph()
    G.add_nodes_from(range(1,13))
    for u,v,_ in EDGES:
        G.add_edge(u,v)

    for u,v,_ in EDGES:
        r = T[(u,v)] / CAP[(u,v)]
        if   (u,v) in rset or (v,u) in rset: col,w,s='#9b59b6',4.0,'dashed'
        elif r >= THRESH:                     col,w,s='#e74c3c',3.5,'solid'
        elif r >= 0.60:                       col,w,s='#f39c12',2.2,'solid'
        else:                                 col,w,s='#27ae60',1.8,'solid'
        nx.draw_networkx_edges(G, POS, edgelist=[(u,v)], ax=ax,
                               edge_color=[col], width=w, style=s, alpha=0.92)

    nc = []
    for n in range(1,13):
        if n == attacked:
            nc.append('#e74c3c'); continue
        nbs = list(G.neighbors(n))
        vals = []
        for nb in nbs:
            k = (n,nb) if (n,nb) in CAP else (nb,n)
            vals.append(T[k]/CAP[k])
        avg = np.mean(vals) if vals else 0
        nc.append('#e74c3c' if avg>=0.85 else '#f39c12' if avg>=0.65 else '#27ae60')

    nx.draw_networkx_nodes(G, POS, ax=ax, node_color=nc, node_size=700,
                           alpha=0.96, linewidths=1.5, edgecolors='#ffffff44')
    nx.draw_networkx_labels(G, POS, ax=ax, labels=LABELS,
                            font_size=7, font_color='white', font_weight='bold')

    el = {(u,v): f"{T[(u,v)]}/{CAP[(u,v)]}" for u,v,_ in EDGES}
    nx.draw_networkx_edge_labels(G, POS, edge_labels=el, ax=ax,
                                 font_size=6.5, font_color='#eee',
                                 bbox=dict(boxstyle='round,pad=0.15',
                                           fc='#1a1a2e', alpha=0.85))

    ax.legend(handles=[
        mpatches.Patch(color='#e74c3c', label='Congested >80%'),
        mpatches.Patch(color='#f39c12', label='Warning 60–80%'),
        mpatches.Patch(color='#27ae60', label='Healthy <60%'),
        mpatches.Patch(color='#9b59b6', label='Rerouted'),
    ], loc='lower right', fontsize=8, framealpha=0.8,
       facecolor='#0e1117', edgecolor='#444', labelcolor='white')

    cg = len(congested(T))
    hl = round((1 - cg/len(CAP))*100, 1)
    ax.set_title(
        f"{title}   |   {'🟢 All Healthy!' if cg==0 else f'🔴 {cg} congested'}   |   Health {hl}%",
        fontsize=11, fontweight='bold', color='white', pad=10)
    ax.axis('off')
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────
#  SESSION STATE — initialise exactly once
# ──────────────────────────────────────────────────────

if 'T' not in st.session_state:
    T0, B0 = make_traffic(seed=42)
    st.session_state.T           = T0
    st.session_state.BASE        = B0
    st.session_state.ep          = 0
    st.session_state.logs        = []
    st.session_state.attacked    = None
    st.session_state.rerouted    = []
    st.session_state.ai_on       = False   # AI auto-heal running flag
    st.session_state.ai_thresh   = 60      # default threshold


# ──────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────

atk = st.session_state.attacked
badge = (f'<span style="background:#3a0000;border:1px solid #e74c3c;border-radius:7px;padding:3px 10px;color:#ff7070;font-weight:700;font-size:12px;margin-left:10px">🔴 ATTACK gNB{atk}</span>' if atk
         else '<span style="background:#071a0e;border:1px solid #27ae60;border-radius:7px;padding:3px 10px;color:#2ecc71;font-weight:700;font-size:12px;margin-left:10px">🟢 NORMAL</span>')
st.markdown(f'<h2 style="margin:0;padding:4px 0">📡 5G Self-Organizing Network {badge}</h2>', unsafe_allow_html=True)
st.caption("Auto Reroute · Attack · Heal · AI Auto Heal · Manual Congestion · Heatmap")
st.divider()

# ──────────────────────────────────────────────────────
#  SIDEBAR  — exactly 4 features, nothing else
# ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎮 Controls")

    # ── Reset ────────────────────────────────────────────
    if st.button("🔄 Reset Network", use_container_width=True):
        T0, B0 = make_traffic(seed=random.randint(0,9999))
        st.session_state.T=T0; st.session_state.BASE=B0
        st.session_state.ep=0; st.session_state.logs=[]
        st.session_state.attacked=None; st.session_state.rerouted=[]
        st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 Auto Reroute")
    if st.button("▶ Run 1 Episode", use_container_width=True, type="primary"):
        # Step 1: Reroute via Dijkstra
        T_new, rer, log = do_reroute(st.session_state.T)
        # Step 2: Restore all still-congested links to base → turns nodes GREEN
        for (u,v) in congested(T_new):
            base_val = st.session_state.BASE.get((u,v),
                       st.session_state.BASE.get((v,u), T_new.get((u,v),5)))
            T_new[(u,v)] = max(5, min(CAP[(u,v)], base_val))
        T_new = safe(T_new)
        # Step 3: Update state
        st.session_state.T        = T_new
        st.session_state.rerouted = rer
        st.session_state.ep      += 1
        st.session_state.attacked = None
        for cls, msg in log:
            st.session_state.logs.insert(0,{"cls":cls,"msg":msg,"ep":st.session_state.ep})
        st.rerun()

    st.markdown("---")
    st.markdown("### 💀 Attack")
    atk_node = st.selectbox("Node", list(range(1,13)), format_func=lambda x:f"gNB{x}")
    if st.button("💥 ATTACK", use_container_width=True):
        st.session_state.T=do_attack(st.session_state.T,atk_node)
        st.session_state.attacked=atk_node; st.session_state.rerouted=[]
        st.session_state.logs.insert(0,{"cls":"bad","ep":st.session_state.ep,
            "msg":f"💀 ATTACK gNB{atk_node} — all links → 100%!"})
        st.rerun()

    st.markdown("---")
    st.markdown("### 🛡️ Heal")
    # Heal specific node
    heal_node = st.selectbox("Heal node", list(range(1,13)), format_func=lambda x:f"gNB{x}", key="heal_sel")
    if st.button("🛡️ Heal This Node", use_container_width=True):
        st.session_state.T=do_heal_node(st.session_state.T, st.session_state.BASE, heal_node)
        if st.session_state.attacked == heal_node:
            st.session_state.attacked = None
        st.session_state.rerouted=[]
        st.session_state.logs.insert(0,{"cls":"info","ep":st.session_state.ep,
            "msg":f"🛡️ gNB{heal_node} healed — its links restored to base"})
        st.rerun()
    # Heal entire network
    if st.button("🛡️ HEAL ALL NETWORK", use_container_width=True):
        st.session_state.T=do_heal(st.session_state.BASE)
        st.session_state.attacked=None; st.session_state.rerouted=[]
        st.session_state.logs.insert(0,{"cls":"info","ep":st.session_state.ep,
            "msg":"🛡️ Full network healed — all links restored"})
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧠 AI Auto Heal")
    ai_thresh = st.slider("Trigger threshold %", 30, 90,
                           st.session_state.ai_thresh, 5,
                           help="AI heals automatically when health falls below this")
    st.session_state.ai_thresh = ai_thresh

    # Toggle button label changes based on state
    ai_label = "🔴 STOP AI Auto Heal" if st.session_state.ai_on else "🟢 START AI Auto Heal"
    if st.button(ai_label, use_container_width=True):
        st.session_state.ai_on = not st.session_state.ai_on
        st.rerun()

    if st.session_state.ai_on:
        st.success("🧠 AI is watching... auto-heals when health < threshold")
    else:
        st.caption("Click START to enable automatic healing")

    st.markdown("---")
    st.markdown("### ⚠️ Manual Link")
    edge_list=[f"{u}↔{v}" for u,v,_ in EDGES]
    sel=st.selectbox("Link",edge_list)
    su,sv=map(int,sel.split("↔"))
    c1,c2=st.columns(2)
    if c1.button("🔴 Congest",use_container_width=True):
        st.session_state.T=do_congest(st.session_state.T,su,sv)
        st.session_state.rerouted=[]
        st.session_state.logs.insert(0,{"cls":"bad","ep":st.session_state.ep,
            "msg":f"⚠️ {su}↔{sv} congested"})
        st.rerun()
    if c2.button("🟢 Clear",use_container_width=True):
        st.session_state.T=do_clear(st.session_state.T,su,sv)
        st.session_state.rerouted=[]
        st.session_state.logs.insert(0,{"cls":"ok","ep":st.session_state.ep,
            "msg":f"✅ {su}↔{sv} cleared"})
        st.rerun()


# ──────────────────────────────────────────────────────
#  AI AUTO HEAL LOOP — runs every render when enabled
#  This is what makes it truly automatic:
#  Every time Streamlit rerenders, it checks health.
#  If health < threshold, it heals and calls st.rerun()
#  which keeps the loop going continuously.
# ──────────────────────────────────────────────────────

import time as _time

if st.session_state.get('ai_on', False):
    cg_check   = congested(st.session_state.T)
    cur_health = round((1 - len(cg_check)/len(CAP))*100, 1)
    if cg_check and cur_health < st.session_state.get('ai_thresh', 60):

        # Step 1: Reroute congested links via Dijkstra
        T_new, rer, log = do_reroute(st.session_state.T)

        # Step 2: For every congested link, also restore it to base
        # This is what actually turns the RED nodes GREEN
        # Pure rerouting only moves overflow — the link stays near capacity
        # Restoring to base physically drops the traffic → node turns green
        for (u, v) in cg_check:
            base_val = st.session_state.BASE.get((u,v),
                       st.session_state.BASE.get((v,u), T_new.get((u,v), 5)))
            T_new[(u,v)] = max(5, min(CAP[(u,v)], base_val))

        # Step 3: Also clear the attacked node flag
        st.session_state.T        = safe(T_new)
        st.session_state.rerouted = rer
        st.session_state.ep      += 1
        st.session_state.attacked = None   # remove red attack banner

        st.session_state.logs.insert(0, {
            "cls":"info", "ep": st.session_state.ep,
            "msg": f"🧠 AI HEALED — health was {cur_health}% → restored all red nodes to GREEN"
        })
        for cls, msg in log:
            st.session_state.logs.insert(0, {
                "cls": cls, "ep": st.session_state.ep, "msg": f"🧠 {msg}"})
        _time.sleep(0.8)
        st.rerun()

# ──────────────────────────────────────────────────────
#  METRICS
# ──────────────────────────────────────────────────────

cg = len(congested(st.session_state.T))
hl = round((1 - cg/len(CAP))*100, 1)
au = round(np.mean([st.session_state.T[k]/CAP[k] for k in CAP])*100, 1)

m1, m2, m3, m4 = st.columns(4)
m1.metric("📡 Episodes",   st.session_state.ep)
m2.metric("⚠️ Congested",  cg, delta="✅ None!" if cg==0 else f"🔴 {cg} links")
m3.metric("💚 Health",     f"{hl}%")
m4.metric("📊 Avg Util",   f"{au}%")

if st.session_state.attacked:
    st.markdown(
        f"<div class='atk'>💀 ATTACK on gNB{st.session_state.attacked} active — "
        f"click ▶ Run 1 Episode or 🛡️ HEAL ALL</div>",
        unsafe_allow_html=True)

st.divider()


# ──────────────────────────────────────────────────────
#  MAP + LOG
# ──────────────────────────────────────────────────────

col_map, col_log = st.columns([2, 1])

with col_map:
    st.markdown("#### 🗺️ Network Map")
    fig = draw(
        st.session_state.T,
        title=f"Episode {st.session_state.ep}",
        attacked=st.session_state.attacked,
        rerouted=st.session_state.rerouted,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Link labels = traffic/capacity  (e.g. 75/100 = 75% load)")

with col_log:
    st.markdown("#### 📋 Log")
    if not st.session_state.logs:
        st.info("Try this:\n\n"
                "1. **💥 ATTACK** any node → links go RED\n"
                "2. **▶ Run 1 Episode** → agent fixes it\n"
                "3. **🛡️ HEAL ALL** → everything GREEN")
    else:
        parts = ['<div class="scr">']
        for e in st.session_state.logs[:40]:
            parts.append(
                f'<div class="log-{e["cls"]}">'
                f'<b>Ep{e["ep"]}</b> — {e["msg"]}</div>')
        parts.append('</div>')
        st.markdown(''.join(parts), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
#  BOTTOM TABS — Bar Chart + Heatmap
# ──────────────────────────────────────────────────────

st.divider()
tab_bar, tab_heat = st.tabs(["📊 Link Utilization", "🌡️ Heatmap"])

with tab_bar:
    df = pd.DataFrame(
        [(f"{u}↔{v}", round(st.session_state.T[(u,v)]/CAP[(u,v)]*100, 1))
         for u,v,_ in EDGES],
        columns=['Link', 'Util%']
    ).sort_values('Util%', ascending=False)
    st.bar_chart(df.set_index('Link'), color="#27ae60", height=220)
    st.caption("Each bar = current utilization % of that link. Red zone = above 80%.")

with tab_heat:
    fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
    fig_hm.patch.set_facecolor('#0e1117')
    ax_hm.set_facecolor('#0e1117')
    cmap = plt.cm.RdYlGn_r

    G_hm = nx.Graph()
    G_hm.add_nodes_from(range(1,13))
    for u,v,_ in EDGES:
        G_hm.add_edge(u,v)

    # Draw thick colored edges
    for u,v,_ in EDGES:
        ratio = st.session_state.T[(u,v)] / CAP[(u,v)]
        col   = cmap(min(ratio, 1.0))
        x1,y1 = POS[u]; x2,y2 = POS[v]
        ax_hm.plot([x1,x2],[y1,y2], color=col, linewidth=8,
                   alpha=0.90, solid_capstyle='round')
        # Label: clean percentage
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax_hm.text(mx, my, f"{int(min(ratio,1.0)*100)}%",
                   ha='center', va='center', fontsize=7,
                   color='white', fontweight='bold',
                   bbox=dict(fc='#111827', alpha=0.80,
                             pad=1.5, boxstyle='round'))

    # Draw nodes colored by worst link
    for n in range(1,13):
        nbs = list(G_hm.neighbors(n))
        vals = [st.session_state.T[(n,nb) if (n,nb) in CAP else (nb,n)] /
                CAP[(n,nb) if (n,nb) in CAP else (nb,n)] for nb in nbs]
        worst = max(vals) if vals else 0
        col   = cmap(min(worst, 1.0))
        x, y  = POS[n]
        ax_hm.scatter(x, y, s=720, color=col, zorder=5,
                      edgecolors='white', linewidths=1.4)
        ax_hm.text(x, y, LABELS[n], ha='center', va='center',
                   fontsize=7, color='white',
                   fontweight='bold', zorder=6)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))
    sm.set_array([])
    cb = fig_hm.colorbar(sm, ax=ax_hm, fraction=0.03, pad=0.02)
    cb.set_label('Utilization', color='white', fontsize=9)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    ax_hm.set_title('🌡️  Network Utilization Heatmap  |  Red = Overloaded · Green = Healthy',
                    fontsize=11, fontweight='bold', color='white', pad=10)
    ax_hm.axis('off')
    plt.tight_layout()
    st.pyplot(fig_hm, use_container_width=True)
    plt.close(fig_hm)
    st.caption("Color intensity shows exact utilization. Numbers = % load on each link.")

st.divider()
st.markdown(
    "<div style='text-align:center;color:#444;font-size:12px'>"
    "5G SON · Auto Rerouting · AI Auto Heal · B.Tech Demo</div>",
    unsafe_allow_html=True)
