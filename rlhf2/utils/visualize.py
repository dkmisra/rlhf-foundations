"""Live training dashboard using Plotly Dash (auto-opens in the browser)."""

from __future__ import annotations

import threading
import time
import webbrowser
from dataclasses import dataclass
from typing import Any

import torch

from utils.data_types import VisualizeConfig

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

# (graph_id, [(series_key, legend_label, color), ...], chart_title)
METRIC_CHARTS: list[tuple[str, list[tuple[str, str, str]], str]] = [
    ("chart-sft-loss", [("sft/loss", "SFT loss", "#2563eb")], "SFT loss"),
    ("chart-rl-total-loss", [("rl/total_loss", "RL total loss", "#db2777")], "RL total loss"),
    # Keep dom id chart-rl-grpo-loss stable (browser/callback ids); series key is rl/rlhf_loss.
    ("chart-rl-grpo-loss", [("rl/rlhf_loss", "RL RLHF loss", "#e11d48")], "RL RLHF loss"),
    ("chart-rl-kl-loss", [("rl/kl_loss", "RL KL loss", "#f472b6")], "RL KL loss"),
    ("chart-sft-grad", [("sft/grad_norm", "SFT grad norm", "#3b82f6")], "SFT gradient norm"),
    ("chart-rl-grad", [("rl/grad_norm", "RL grad norm", "#10b981")], "RL gradient norm"),
    ("chart-rl-reward", [("rl/mean_reward", "Train reward", "#059669")], "RL mean reward (train)"),
    ("chart-rl-entropy", [("rl/avg_entropy", "Entropy", "#7c3aed")], "RL avg entropy (response)"),
    ("chart-sft-eval", [("sft/eval_reward", "Eval reward", "#0284c7")], "SFT eval reward"),
    (
        "chart-sft-eval-nll",
        [("sft/eval_teacher_nll", "Teacher NLL", "#6366f1")],
        "SFT eval teacher NLL",
    ),
    ("chart-rl-eval", [("rl/eval_reward", "Eval reward", "#0d9488")], "RL eval reward"),
]
CHART_IDS = [spec[0] for spec in METRIC_CHARTS]


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

    def __init__(self, config: VisualizeConfig):
        self.config = config
        self._lock = threading.Lock()
        self._series: dict[str, list[tuple[int, float]]] = {}
        self._generations: list[GenerationSample] = []
        self._step_counter = {"sft": 0, "rl": 0}
        self._server_thread: threading.Thread | None = None
        self._started = False

        if self.config.enabled:
            self._start_dashboard()

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

    def log_scalar(self, name: str, value: float, step: int) -> None:
        if not self.config.enabled:
            return
        with self._lock:
            self._series.setdefault(name, []).append((step, float(value)))

    def log_sft_step(
        self,
        loss: float,
        grad_norm_value: float,
        *,
        step: int | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        step = step if step is not None else self._next_step("sft")
        if step % self.config.log_every != 0:
            return
        self.log_scalar("sft/loss", loss, step)
        self.log_scalar("sft/grad_norm", grad_norm_value, step)

    def log_rl_step(
        self,
        *,
        step: int | None = None,
        total_loss: float,
        rlhf_loss: float,
        kl_loss: float,
        mean_reward: float,
        grad_norm_value: float,
        avg_entropy: float | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        step = step if step is not None else self._next_step("rl")
        if step % self.config.log_every != 0:
            return
        self.log_scalar("rl/total_loss", total_loss, step)
        self.log_scalar("rl/rlhf_loss", rlhf_loss, step)
        self.log_scalar("rl/kl_loss", kl_loss, step)
        self.log_scalar("rl/mean_reward", mean_reward, step)
        self.log_scalar("rl/grad_norm", grad_norm_value, step)
        if avg_entropy is not None:
            self.log_scalar("rl/avg_entropy", avg_entropy, step)

    def log_eval_reward(self, reward: float, phase: str, step: int) -> None:
        self.log_scalar(f"{phase}/eval_reward", reward, step)

    def log_eval_teacher_nll(self, nll: float, step: int) -> None:
        self.log_scalar("sft/eval_teacher_nll", nll, step)

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
            for prompt, completion, reward, target in zip(
                prompts, completions, rewards, targets
            ):
                self._generations.append(
                    GenerationSample(
                        step=step,
                        phase=phase,
                        prompt=prompt,
                        completion=completion,
                        reward=reward,
                        target=target,
                    )
                )
            excess = len(self._generations) - self.config.max_generation_samples
            if excess > 0:
                self._generations = self._generations[excess:]

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

    def _next_step(self, phase: str) -> int:
        with self._lock:
            self._step_counter[phase] += 1
            return self._step_counter[phase]

    def snapshot(self) -> tuple[dict[str, list[tuple[int, float]]], list[GenerationSample]]:
        with self._lock:
            series = {k: list(v) for k, v in self._series.items()}
            generations = list(self._generations)
        return series, generations

    def _run_dash(self) -> None:
        metric_charts = list(METRIC_CHARTS)
        chart_ids = [chart_id for chart_id, _, _ in metric_charts]

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
                    style={"marginBottom": "20px"},
                    children=[
                        html.H1(
                            "RLHF Training Monitor",
                            style={"margin": 0, "fontWeight": 700, "letterSpacing": "-0.02em"},
                        ),
                        html.P(
                            "Live metrics (one chart per series) and sampled generations.",
                            style={"margin": "8px 0 0", "color": MUTED_COLOR},
                        ),
                    ],
                ),
                dcc.Interval(id="refresh", interval=1500, n_intervals=0),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(2, minmax(320px, 1fr))",
                        "gap": "16px",
                        "marginBottom": "24px",
                    },
                    children=[
                        dcc.Graph(id=chart_id, style={"height": "260px"})
                        for chart_id in chart_ids
                    ],
                ),
                html.H3("Recent generations", style={"marginTop": "8px"}),
                html.Div(id="generations-table"),
            ],
        )

        # Single multi-output callback (Dash batches all chart outputs in one request).
        chart_outputs = [Output(chart_id, "figure") for chart_id in chart_ids]
        table_output = Output("generations-table", "children")

        @app.callback(
            chart_outputs + [table_output],
            Input("refresh", "n_intervals"),
            prevent_initial_call=False,
        )
        def _update_all(_n: int):
            series, generations = self.snapshot()
            figures = [
                self._build_line_chart(series, series_specs, title)
                for _, series_specs, title in metric_charts
            ]
            return figures + [self._build_generations_table(generations)]

        app.run(
            host=self.config.host,
            port=self.config.port,
            debug=False,
            use_reloader=False,
        )

    def _build_line_chart(
        self,
        series: dict[str, list[tuple[int, float]]],
        series_specs: list[tuple[str, str, str]],
        title: str,
    ) -> go.Figure:
        fig = go.Figure()
        has_data = False
        for key, label, color in series_specs:
            points = series.get(key, [])
            if not points:
                continue
            has_data = True
            steps, values = zip(*points)
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=values,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=5, color=color),
                )
            )
        if not has_data:
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
            showlegend=len(series_specs) > 1,
            hovermode="x unified",
        )
        fig.update_xaxes(title_text="step", gridcolor=GRID_COLOR, linecolor=BORDER_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR, linecolor=BORDER_COLOR)
        return fig

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"{r}, {g}, {b}"

    def _build_generations_table(self, generations: list[GenerationSample]) -> Any:
        if not generations:
            return html.P("No generations logged yet.", style={"color": MUTED_COLOR})

        rows = [
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

        return dash_table.DataTable(
            data=rows,
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


def create_visualizer(config: VisualizeConfig | None) -> TrainingVisualizer | None:
    if config is None or not config.enabled:
        return None
    return TrainingVisualizer(config)
