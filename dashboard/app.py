"""
FRIDAY-MAF Live Dashboard
Streamlit dashboard providing full observability of the multi-agent system.
Panels: Task Input | Live Monitor | Event Stream | DAG View | Task Detail | System Health
"""
from __future__ import annotations
import time
import json
import requests
import streamlit as st
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE    = "http://localhost:8000"
REFRESH_SEC = 3

st.set_page_config(
    page_title = "FRIDAY-MAF Dashboard",
    page_icon  = "🤖",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .sub-header  { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .status-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em;
    }
    .status-CREATED   { background:#e3f2fd; color:#1565c0; }
    .status-PLANNING  { background:#fff8e1; color:#f57f17; }
    .status-EXECUTING { background:#e8f5e9; color:#2e7d32; }
    .status-REVIEWING { background:#f3e5f5; color:#6a1b9a; }
    .status-RETRYING  { background:#fff3e0; color:#e65100; }
    .status-COMPLETED { background:#e8f5e9; color:#1b5e20; }
    .status-FAILED    { background:#ffebee; color:#b71c1c; }
    .event-row { font-size: 0.8rem; padding: 4px 0; border-bottom: 1px solid #f0f0f0; }
    .metric-card { background:#f8f9fa; padding:12px; border-radius:8px; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(path: str, params: dict = None) -> Optional[Any]:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _post(path: str, body: dict) -> Optional[Any]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def status_badge(status: str) -> str:
    return f'<span class="status-badge status-{status}">{status}</span>'


def fmt_ts(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ts[:19] if ts else "—"


EVENT_ICONS = {
    "TASK_CREATED":   "🟢",
    "TASK_ASSIGNED":  "🔵",
    "TASK_COMPLETED": "✅",
    "TASK_FAILED":    "❌",
    "CRITIC_FAILED":  "⚠️",
    "RETRY_TRIGGERED":"🔄",
    "PLAN_COMPLETED": "📋",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 FRIDAY-MAF")
    st.markdown("Multi-Agent Framework")
    st.divider()

    health = _get("/health")
    if health:
        col1, col2 = st.columns(2)
        col1.metric("Provider", health.get("provider", "—").upper())
        col2.metric("Model",    health.get("model",    "—"))

        active = health.get("active_tasks", 0)
        loop   = "✅ Running" if health.get("loop_running") else "⛔ Stopped"
        st.metric("Active Tasks", active)
        st.caption(f"Autonomy Loop: {loop}")

        stats = health.get("task_stats", {})
        if stats:
            st.divider()
            st.caption("Task Stats")
            for s, cnt in stats.items():
                st.write(f"{status_badge(s)} × {cnt}", unsafe_allow_html=True)
    else:
        st.error("⚠️ API offline — start the backend first.")

    st.divider()
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_sec  = st.slider("Interval (s)", 1, 10, REFRESH_SEC)

    if st.button("🔄 Refresh now", use_container_width=True):
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🤖 FRIDAY-MAF Live Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Autonomous Multi-Agent Execution Engine</p>', unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_submit, tab_monitor, tab_events, tab_dag, tab_detail, tab_health = st.tabs([
    "📤 Submit Task",
    "📊 Live Monitor",
    "📡 Event Stream",
    "🗺️ DAG View",
    "🔍 Task Detail",
    "🏥 System Health",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — TASK INPUT
# ────────────────────────────────────────────────────────────────────────────

with tab_submit:
    st.subheader("Submit a New Task")
    st.caption("Tasks are picked up by the autonomy loop and executed in the background.")

    examples = [
        "Research the latest advances in quantum computing",
        "Write a Python function to sort a list of dictionaries by a key",
        "Analyse the growth trend of global renewable energy adoption",
        "Explain the differences between TCP and UDP protocols",
    ]

    selected = st.selectbox("Quick example:", ["(type your own)"] + examples)
    default  = "" if selected == "(type your own)" else selected

    with st.form("task_form"):
        task_input = st.text_area(
            "Task description",
            value    = default,
            height   = 120,
            placeholder = "Describe what you want the agents to do…",
        )
        submitted = st.form_submit_button("🚀 Submit Task", use_container_width=True, type="primary")

    if submitted and task_input.strip():
        with st.spinner("Submitting…"):
            resp = _post("/task", {"message": task_input.strip()})
        if resp and "error" not in resp:
            st.success(f"✅ Task submitted! ID: `{resp['task_id']}`")
            st.info(f"Status: **{resp['status']}** — switch to **Live Monitor** to track progress.")
            if "last_task_id" not in st.session_state:
                st.session_state.last_task_id = resp["task_id"]
            st.session_state.last_task_id = resp["task_id"]
        else:
            st.error(f"❌ Submit failed: {resp}")
    elif submitted:
        st.warning("Please enter a task description.")

    if "last_task_id" in st.session_state:
        st.divider()
        st.caption(f"Last submitted task: `{st.session_state.last_task_id}`")


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — LIVE MONITOR
# ────────────────────────────────────────────────────────────────────────────

with tab_monitor:
    st.subheader("Live Task Monitor")

    status_filter = st.selectbox(
        "Filter by status",
        ["ALL", "CREATED", "PLANNING", "EXECUTING", "REVIEWING",
         "RETRYING", "COMPLETED", "FAILED"],
        key="monitor_filter",
    )

    params  = {} if status_filter == "ALL" else {"status": status_filter}
    tasks   = _get("/tasks", params) or []

    if not tasks:
        st.info("No tasks yet — submit one in the **Submit Task** tab.")
    else:
        st.caption(f"{len(tasks)} task(s) shown")

        for t in tasks:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                c1.markdown(f"**{t['goal'][:80]}{'…' if len(t['goal']) > 80 else ''}**")
                c2.markdown(status_badge(t["status"]), unsafe_allow_html=True)
                c3.caption(f"🧩 {t['subtask_count']} subtasks")
                c4.caption(fmt_ts(t["updated_at"]))

                st.caption(f"`{t['task_id']}`  ·  created {fmt_ts(t['created_at'])}")

                if t.get("error"):
                    st.error(f"Error: {t['error'][:200]}")


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — EVENT STREAM
# ────────────────────────────────────────────────────────────────────────────

with tab_events:
    st.subheader("Agent Event Stream")

    col_filter, col_limit = st.columns([2, 1])
    task_filter = col_filter.text_input("Filter by task ID (optional)", key="event_task_filter")
    evt_limit   = col_limit.number_input("Max events", min_value=10, max_value=500, value=100, step=10)

    ev_params = {"limit": evt_limit}
    if task_filter.strip():
        ev_params["task_id"] = task_filter.strip()

    events = _get("/events", ev_params) or []

    if not events:
        st.info("No events yet.")
    else:
        st.caption(f"{len(events)} events (newest first)")

        for ev in events:
            icon    = EVENT_ICONS.get(ev.get("event_type", ""), "📌")
            ts      = fmt_ts(ev.get("timestamp", ""))
            etype   = ev.get("event_type", "UNKNOWN")
            task_id = ev.get("task_id", "")[:8]
            payload = ev.get("payload", {})

            with st.container():
                c1, c2, c3 = st.columns([1, 2, 4])
                c1.caption(ts)
                c2.markdown(f"{icon} **{etype}**")
                c3.caption(f"`{task_id}…`  {json.dumps(payload)[:80]}")
            st.divider()


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — DAG VISUALIZATION
# ────────────────────────────────────────────────────────────────────────────

with tab_dag:
    st.subheader("Task DAG Visualization")

    tasks_for_dag = _get("/tasks") or []
    task_ids      = {f"{t['goal'][:40]} ({t['task_id'][:8]})": t["task_id"] for t in tasks_for_dag}

    if not task_ids:
        st.info("No tasks available yet.")
    else:
        chosen_label = st.selectbox("Select task to visualize", list(task_ids.keys()), key="dag_select")
        chosen_id    = task_ids[chosen_label]

        detail = _get(f"/task/{chosen_id}/detail")
        if detail:
            task_data = detail.get("task", {})
            subtasks  = task_data.get("subtasks", [])

            if not subtasks:
                st.info("This task has no subtasks yet (planning may be in progress).")
            else:
                st.caption(f"Goal: **{task_data.get('goal', '')}**")
                st.caption(f"Overall status: ")
                st.markdown(status_badge(task_data.get("status", "UNKNOWN")), unsafe_allow_html=True)
                st.divider()

                STATUS_COLOR = {
                    "CREATED":   "🔘",
                    "PLANNING":  "🟡",
                    "EXECUTING": "🔵",
                    "REVIEWING": "🟣",
                    "RETRYING":  "🟠",
                    "COMPLETED": "🟢",
                    "FAILED":    "🔴",
                }

                # Build dependency graph display
                id_to_idx = {st_data["task_id"]: i for i, st_data in enumerate(subtasks)}

                for i, st_data in enumerate(subtasks):
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([1, 4, 2])
                        icon   = STATUS_COLOR.get(st_data.get("status", "CREATED"), "⚪")
                        deps   = st_data.get("dependencies", [])
                        dep_str = ", ".join(
                            f"#{id_to_idx.get(d, '?')+1}" for d in deps
                        ) if deps else "none"

                        c1.markdown(f"### {icon}")
                        c2.markdown(f"**#{i+1}** {st_data.get('task_description', '')[:80]}")
                        c2.caption(f"Agent: `{st_data.get('required_agent', '')}` · deps: {dep_str}")
                        c3.markdown(status_badge(st_data.get("status", "CREATED")), unsafe_allow_html=True)

                        if st_data.get("retry_count", 0) > 0:
                            c3.caption(f"⚠️ {st_data['retry_count']} retries")

                        if st_data.get("result"):
                            with st.expander("View result"):
                                result = st_data["result"]
                                if isinstance(result, dict):
                                    st.json(result)
                                else:
                                    st.write(result)

                        if st_data.get("error"):
                            st.error(f"Error: {st_data['error'][:200]}")


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — TASK DETAIL
# ────────────────────────────────────────────────────────────────────────────

with tab_detail:
    st.subheader("Task Detail View")

    detail_id = st.text_input(
        "Task ID",
        value  = st.session_state.get("last_task_id", ""),
        placeholder = "Paste a task_id here…",
        key    = "detail_task_id",
    )

    if detail_id.strip():
        detail = _get(f"/task/{detail_id.strip()}/detail")

        if detail is None:
            st.error("Task not found or API offline.")
        else:
            task_data = detail.get("task", {})
            events    = detail.get("events", [])

            # ── Overview ───────────────────────────────────────────────────
            st.markdown(f"### {task_data.get('goal', '—')}")
            st.markdown(status_badge(task_data.get("status", "UNKNOWN")), unsafe_allow_html=True)
            st.caption(f"Input: *{task_data.get('user_input', '')}*")

            if task_data.get("error"):
                st.error(f"**Error:** {task_data['error']}")

            # ── Final result ───────────────────────────────────────────────
            if task_data.get("final_result"):
                st.divider()
                st.markdown("#### 📝 Final Result")
                result = task_data["final_result"]
                if isinstance(result, dict):
                    st.json(result)
                elif isinstance(result, str):
                    st.markdown(result)
                else:
                    st.write(result)

            # ── Planner output / Subtasks ─────────────────────────────────
            subtasks = task_data.get("subtasks", [])
            if subtasks:
                st.divider()
                st.markdown(f"#### 🧩 Planner Output ({len(subtasks)} subtasks)")
                for i, st_data in enumerate(subtasks):
                    with st.expander(
                        f"#{i+1} [{st_data.get('status','?')}] "
                        f"{st_data.get('task_description','')[:60]}",
                        expanded=False,
                    ):
                        cols = st.columns(3)
                        cols[0].metric("Agent",   st_data.get("required_agent", "—"))
                        cols[1].metric("Status",  st_data.get("status", "—"))
                        cols[2].metric("Retries", st_data.get("retry_count", 0))

                        if st_data.get("result"):
                            st.markdown("**Worker Output:**")
                            r = st_data["result"]
                            if isinstance(r, dict):
                                st.json(r)
                            else:
                                st.write(r)

                        if st_data.get("error"):
                            st.error(f"Error: {st_data['error']}")

            # ── Events ────────────────────────────────────────────────────
            if events:
                st.divider()
                st.markdown(f"#### 📡 Events ({len(events)})")
                for ev in events[:50]:
                    icon  = EVENT_ICONS.get(ev.get("event_type", ""), "📌")
                    ts    = fmt_ts(ev.get("timestamp", ""))
                    etype = ev.get("event_type", "")
                    st.caption(f"{icon} `{ts}` **{etype}** — {json.dumps(ev.get('payload',{}))[:100]}")

            # ── Memory metadata ────────────────────────────────────────────
            meta = task_data.get("metadata", {})
            if meta:
                st.divider()
                st.markdown("#### 🧠 Task Metadata")
                st.json(meta)
    else:
        st.info("Enter a task ID above, or submit a task in the **Submit Task** tab.")


# ────────────────────────────────────────────────────────────────────────────
# TAB 6 — SYSTEM HEALTH
# ────────────────────────────────────────────────────────────────────────────

with tab_health:
    st.subheader("System Health")

    health = _get("/health")

    if health is None:
        st.error("❌ Cannot reach API at " + API_BASE)
        st.code("Make sure the FastAPI server is running on port 5000")
    else:
        # Top-level status
        status_ok = health.get("status") == "ok"
        st.markdown(
            "### " + ("✅ System Operational" if status_ok else "⚠️ System Degraded")
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("API",          "✅ Online")
        col2.metric("Autonomy Loop", "✅ Running" if health.get("loop_running") else "⛔ Stopped")
        col3.metric("Active Tasks",  health.get("active_tasks", 0))
        col4.metric("Events Stored", health.get("event_count", 0))

        st.divider()

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### 📊 Task Statistics")
            stats = health.get("task_stats", {})
            if stats:
                for s, cnt in stats.items():
                    st.progress(
                        min(cnt / max(sum(stats.values()), 1), 1.0),
                        text=f"{s}: {cnt}",
                    )
            else:
                st.caption("No tasks yet.")

            st.divider()
            st.markdown("#### ⚙️ Configuration")
            st.json({
                "provider": health.get("provider"),
                "model":    health.get("model"),
                "version":  health.get("version"),
            })

        with col_r:
            st.markdown("#### 📈 Metrics Counters")
            counters = health.get("metrics", {}).get("counters", {})
            if counters:
                for k, v in sorted(counters.items()):
                    st.write(f"`{k}` → **{v:.0f}**")
            else:
                st.caption("No metrics yet.")

            st.divider()
            st.markdown("#### 🔍 Recent Spans")
            spans = health.get("recent_spans", [])
            if spans:
                for span in spans[:8]:
                    color = "🟢" if span.get("status") == "ok" else "🔴"
                    st.caption(
                        f"{color} `{span.get('name','')}` "
                        f"[{span.get('agent','')}] "
                        f"{span.get('duration_ms', 0):.1f}ms"
                    )
            else:
                st.caption("No spans yet.")

    # Error rate from metrics
    metrics_data = (health or {}).get("metrics", {})
    histograms   = metrics_data.get("histograms", {})
    if histograms:
        st.divider()
        st.markdown("#### 📉 Performance Histograms")
        for k, v in histograms.items():
            st.write(
                f"`{k}` — mean {v.get('mean',0):.1f}ms, "
                f"p95 {v.get('p95',0):.1f}ms, "
                f"count {v.get('count',0)}"
            )


# ── Auto-refresh ──────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
