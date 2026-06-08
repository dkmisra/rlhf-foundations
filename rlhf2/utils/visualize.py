"""Live training dashboard using Plotly Dash (auto-opens in the browser)."""

from __future__ import annotations

import logging
import threading
import time
import webbrowser
from dataclasses import dataclass
from typing import Any

import torch

from rlhf2.utils.data_types import VisualizeConfig

try:
    from dash import Dash, Input, Output, dash_table, dcc, html
    import plotly.graph_objects as go
except ImportError as exc:
    raise ImportError(
        "Visualization requires dash and plotly. Install with: pip install dash plotly"
    ) from exc


PLOTLY_TEMPLATE = "plotly_white"
PAGE_BG = "#f8fafc"
PLOT_BG = "#ffffff"
TEXT_COLOR = "#0f172a"
MUTED_COLOR = "#64748b"
GRID_COLOR = "#e2e8f0"
BORDER_COLOR = "#e2e8f0"

# Per stage-kind chart catalog: each entry is (metric_key, color, chart_title).
# Charts are created statically per stage, so the Dash layout/callback outputs are
# fixed up front (no flashing) even though the number of stages is config-driven.
METRIC_SPECS: dict[str, list[tuple[str, str, str]]] = {
    "sft": [
        ("loss", "#2563eb", "Loss"),
        ("grad_norm", "#3b82f6", "Gradient norm"),
        ("eval_reward", "#0284c7", "Eval reward"),
        ("eval_teacher_nll", "#6366f1", "Eval teacher NLL"),
    ],
    "rl": [
        ("total_loss", "#db2777", "Total loss"),
        ("rlhf_loss", "#e11d48", "RLHF loss"),
        ("kl_loss", "#f472b6", "KL loss"),
        ("grad_norm", "#10b981", "Gradient norm"),
        ("mean_reward", "#059669", "Mean reward (train)"),
        ("eval_reward", "#0d9488", "Eval reward"),
        ("entropy", "#7c3aed", "Avg entropy (response)"),
    ],
}


@dataclass
class GenerationSample:
    step: int
    phase: str
    prompt: str
    completion: str
    reward: float
    target: str | None = None


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.data.norm(2).item() ** 2
    return total**0.5


class TrainingVisualizer:
    """Thread-safe logger + Dash server for live RL/SFT monitoring."""

    def __init__(self, config: VisualizeConfig, stages: list[tuple[str, str]]):
        """:param stages: ordered ``(display_name, kind)`` pairs, kind in {"sft", "rl"}."""
        self.config = config
        self.stages = stages
        self._lock = threading.Lock()
        # Series keyed by f"{stage_idx}/{metric}"; generations bucketed per stage.
        self._series: dict[str, list[tuple[int, float]]] = {}
        self._generations: dict[int, list[GenerationSample]] = {i: [] for i in range(len(stages))}
        self._current_stage = 0
        self._server_thread: threading.Thread | None = None
        self._started = False

        if self.config.enabled:
            self._start_dashboard()

    def begin_stage(self, stage_idx: int) -> None:
        """Mark the active stage; subsequent metric/generation logs route to it."""
        self._current_stage = stage_idx

    def _start_dashboard(self) -> None:
        if self._started:
            return
        self._started = True
        self._server_thread = threading.Thread(
            target=self._run_dash,
            name="rlhf-dashboard",
            daemon=True,
        )
        self._server_thread.start()
        time.sleep(1.2)
        if self._server_thread is not None and not self._server_thread.is_alive():
            print(
                f"Warning: dashboard failed to start on port {self.config.port} "
                "(port may be in use by a previous run). "
                "Stop the old process or change visualize.port, then hard-refresh the browser."
            )
            return
        if self.config.open_browser:
            webbrowser.open(f"http://{self.config.host}:{self.config.port}")

    def block_until_exit(self) -> None:
        """Keep the (daemon) dashboard server alive after training finishes.

        Without this the process exits as soon as training returns, killing the
        server thread so the live page stops responding (e.g. the smoothing
        slider no longer triggers a redraw). Blocks until Ctrl+C.
        """
        if not self.config.enabled or self._server_thread is None:
            return
        print(
            f"Training complete. Dashboard still live at "
            f"http://{self.config.host}:{self.config.port} — press Ctrl+C to exit."
        )
        try:
            while self._server_thread.is_alive():
                self._server_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            print("Shutting down dashboard.")

    def log_metrics(self, step: int, metrics: dict[str, float], *, throttle: bool = True) -> None:
        """Log a set of named scalars for the current stage at ``step``.

        ``throttle`` honors ``log_every`` for high-frequency training metrics;
        eval metrics pass ``throttle=False`` so they are always recorded.
        """
        if not self.config.enabled:
            return
        if throttle and step % self.config.log_every != 0:
            return
        with self._lock:
            for name, value in metrics.items():
                self._series.setdefault(f"{self._current_stage}/{name}", []).append((step, float(value)))

    def log_generations(
        self,
        *,
        step: int,
        phase: str,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
        targets: list[str] | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        if self.config.generation_log_every <= 0:
            return
        if step % self.config.generation_log_every != 0:
            return

        targets = targets or [None] * len(prompts)
        with self._lock:
            bucket = self._generations.setdefault(self._current_stage, [])
            for prompt, completion, reward, target in zip(
                prompts, completions, rewards, targets
            ):
                bucket.append(
                    GenerationSample(
                        step=step,
                        phase=phase,
                        prompt=prompt,
                        completion=completion,
                        reward=reward,
                        target=target,
                    )
                )
            excess = len(bucket) - self.config.max_generation_samples
            if excess > 0:
                del bucket[:excess]

    def log_rollout_batch(
        self,
        *,
        step: int,
        phase: str,
        prompts: list[str],
        metadata_list: list[dict],
        generations_per_prompt: list[list[str]],
        rewards_per_prompt: list[list[float]],
    ) -> None:
        flat_prompts: list[str] = []
        flat_completions: list[str] = []
        flat_rewards: list[float] = []
        flat_targets: list[str | None] = []

        for prompt, metadata, gens, rews in zip(
            prompts, metadata_list, generations_per_prompt, rewards_per_prompt
        ):
            target = metadata.get("target_completion")
            for completion, reward in zip(gens, rews):
                flat_prompts.append(prompt)
                flat_completions.append(completion)
                flat_rewards.append(reward)
                flat_targets.append(target)

        self.log_generations(
            step=step,
            phase=phase,
            prompts=flat_prompts,
            completions=flat_completions,
            rewards=flat_rewards,
            targets=flat_targets,
        )

    def snapshot(self) -> tuple[dict[str, list[tuple[int, float]]], dict[int, list[GenerationSample]]]:
        with self._lock:
            series = {k: list(v) for k, v in self._series.items()}
            generations = {i: list(v) for i, v in self._generations.items()}
        return series, generations

    def _run_dash(self) -> None:
        # Werkzeug logs every HTTP request (e.g. the per-interval POST
        # /_dash-update-component); silence everything below errors.
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        # Flat list of every chart across all stages: (stage_idx, key, color, title, dom_id).
        chart_specs = [
            (idx, key, color, title, f"chart-{idx}-{key}")
            for idx, (_, kind) in enumerate(self.stages)
            for key, color, title in METRIC_SPECS[kind]
        ]

        app = Dash(__name__, suppress_callback_exceptions=True)
        app.title = "RLHF Training Monitor"

        app.layout = html.Div(
            style={
                "backgroundColor": PAGE_BG,
                "color": TEXT_COLOR,
                "minHeight": "100vh",
                "fontFamily": "'Inter', 'Segoe UI', system-ui, sans-serif",
                "padding": "24px",
            },
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "flex-start",
                        "gap": "24px",
                        "marginBottom": "20px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.H1(
                                    "RLHF Training Monitor",
                                    style={
                                        "margin": 0,
                                        "fontWeight": 700,
                                        "letterSpacing": "-0.02em",
                                    },
                                ),
                                html.P(
                                    "Live metrics and sampled generations, grouped by training stage.",
                                    style={"margin": "8px 0 0", "color": MUTED_COLOR},
                                ),
                            ],
                        ),
                        html.Div(
                            style={"width": "220px", "flexShrink": 0},
                            children=[
                                html.Label(
                                    "Smoothing",
                                    style={
                                        "fontSize": "13px",
                                        "fontWeight": 600,
                                        "color": MUTED_COLOR,
                                    },
                                ),
                                dcc.Slider(
                                    id="smoothing-slider",
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    value=0,
                                    marks={0: "0", 0.5: "0.5", 1: "1"},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Interval(id="refresh", interval=1500, n_intervals=0),
                *[
                    self._build_stage_section(idx, name, kind)
                    for idx, (name, kind) in enumerate(self.stages)
                ],
            ],
        )

        # Single multi-output callback (Dash batches all outputs in one request).
        # Tables update only their `data` so pagination/sort state survives refreshes.
        chart_outputs = [Output(spec[4], "figure") for spec in chart_specs]
        table_outputs = [Output(f"gen-table-{idx}", "data") for idx in range(len(self.stages))]

        @app.callback(
            chart_outputs + table_outputs,
            [Input("refresh", "n_intervals"), Input("smoothing-slider", "value")],
            prevent_initial_call=False,
        )
        def _update_all(_n: int, smoothing: float | None):
            series, generations = self.snapshot()
            figures = [
                self._build_line_chart(series.get(f"{idx}/{key}", []), color, title, smoothing or 0.0)
                for idx, key, color, title, _ in chart_specs
            ]
            tables = [self._generation_rows(generations.get(idx, [])) for idx in range(len(self.stages))]
            return figures + tables

        app.run(
            host=self.config.host,
            port=self.config.port,
            debug=False,
            use_reloader=False,
        )

    @staticmethod
    def _smooth(values: tuple[float, ...], weight: float) -> list[float]:
        """TensorBoard-style exponential moving average with debiasing.

        `weight` in [0, 1): higher means smoother. Clamped below 1 so the top of
        the slider stays meaningful instead of collapsing onto the first value.
        """
        weight = min(weight, 0.99)
        smoothed: list[float] = []
        last = 0.0
        num_accum = 0
        for value in values:
            last = last * weight + (1 - weight) * value
            num_accum += 1
            debias = 1 - weight**num_accum
            smoothed.append(last / debias if debias > 0 else value)
        return smoothed

    def _build_line_chart(
        self,
        points: list[tuple[int, float]],
        color: str,
        title: str,
        smoothing: float = 0.0,
    ) -> go.Figure:
        fig = go.Figure()
        if points:
            steps, values = zip(*points)
            if smoothing > 0:
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=values,
                        mode="lines",
                        name=title,
                        line=dict(color=f"rgba({self._hex_to_rgb(color)}, 0.25)", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=self._smooth(values, smoothing),
                        mode="lines",
                        name=title,
                        line=dict(color=color, width=2),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=values,
                        mode="lines+markers",
                        name=title,
                        line=dict(color=color, width=2),
                        marker=dict(size=5, color=color),
                    )
                )
        else:
            fig.add_annotation(
                text="Waiting for data…",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color=MUTED_COLOR),
            )

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=dict(text=title, font=dict(size=14, color=TEXT_COLOR), x=0),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            font=dict(color=TEXT_COLOR, size=12),
            margin=dict(l=48, r=16, t=40, b=40),
            showlegend=False,
            hovermode="x unified",
            # Preserve zoom/pan (and avoid the visible flash) across refreshes.
            uirevision=title,
        )
        fig.update_xaxes(title_text="step", gridcolor=GRID_COLOR, linecolor=BORDER_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR, linecolor=BORDER_COLOR)
        return fig

    def _build_stage_section(self, stage_idx: int, name: str, kind: str) -> Any:
        """A titled block of charts plus a generations table for one stage."""
        charts = [
            dcc.Graph(id=f"chart-{stage_idx}-{key}", style={"height": "260px"})
            for key, _, _ in METRIC_SPECS[kind]
        ]
        return html.Div(
            style={
                "border": f"1px solid {BORDER_COLOR}",
                "borderRadius": "12px",
                "backgroundColor": PLOT_BG,
                "padding": "16px",
                "marginBottom": "24px",
            },
            children=[
                html.H2(
                    f"Stage {stage_idx + 1}: {name}",
                    style={"margin": "0 0 4px", "fontWeight": 700, "letterSpacing": "-0.01em"},
                ),
                html.P(
                    f"{kind.upper()} stage",
                    style={"margin": "0 0 16px", "color": MUTED_COLOR, "fontSize": "13px"},
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(2, minmax(320px, 1fr))",
                        "gap": "16px",
                        "marginBottom": "16px",
                    },
                    children=charts,
                ),
                html.H3("Recent generations", style={"margin": "8px 0"}),
                self._build_generations_table(stage_idx),
            ],
        )

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"{r}, {g}, {b}"

    @staticmethod
    def _generation_rows(generations: list[GenerationSample]) -> list[dict]:
        return [
            {
                "step": g.step,
                "phase": g.phase,
                "prompt": g.prompt,
                "completion": g.completion,
                "reward": f"{g.reward:.2f}",
                "target": g.target or "",
            }
            for g in reversed(generations)
        ]

    def _build_generations_table(self, stage_idx: int) -> Any:
        """Static DataTable created once in the layout; rows are updated in-place
        via the refresh callback so paging/sort state is preserved."""
        return dash_table.DataTable(
            id=f"gen-table-{stage_idx}",
            data=[],
            columns=[
                {"name": "Step", "id": "step"},
                {"name": "Phase", "id": "phase"},
                {"name": "Prompt", "id": "prompt"},
                {"name": "Completion", "id": "completion"},
                {"name": "Reward", "id": "reward"},
                {"name": "Target", "id": "target"},
            ],
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#f1f5f9",
                "color": TEXT_COLOR,
                "fontWeight": "600",
                "border": f"1px solid {BORDER_COLOR}",
            },
            style_cell={
                "backgroundColor": PLOT_BG,
                "color": TEXT_COLOR,
                "border": f"1px solid {BORDER_COLOR}",
                "fontFamily": "ui-monospace, monospace",
                "fontSize": "13px",
                "textAlign": "left",
                "padding": "10px",
                "maxWidth": "320px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": "{reward} = 1.00"},
                    "backgroundColor": "#dcfce7",
                },
            ],
            page_size=10,
        )


def create_visualizer(
    config: VisualizeConfig | None,
    stages: list[tuple[str, str]],
) -> TrainingVisualizer | None:
    if config is None or not config.enabled:
        return None
    return TrainingVisualizer(config, stages)
