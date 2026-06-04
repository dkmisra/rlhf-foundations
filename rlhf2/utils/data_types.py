from typing import Literal

from pydantic import BaseModel, Field, model_validator


class InferenceConfig(BaseModel):
    """Sampling settings used during rollout / evaluation."""

    max_tokens: int = Field(default=40, description="Maximum tokens to generate per completion")
    temp: float = Field(default=1.0, description="Softmax temperature for sampling")


class SFTConfig(BaseModel):
    """Supervised fine-tuning stage run before RL."""

    enabled: bool = True
    max_epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = Field(default=20, description="Run eval every N SFT steps; 0 disables")


class VisualizeConfig(BaseModel):
    """Live training dashboard (Plotly Dash) opened automatically in the browser."""

    enabled: bool = True
    port: int = 8050
    open_browser: bool = True
    host: str = "127.0.0.1"
    log_every: int = Field(default=1, description="Log scalars every N training steps")
    generation_log_every: int = Field(
        default=20,
        description="Append generation samples every N steps (0 disables)",
    )
    max_generation_samples: int = Field(
        default=12,
        description="Max prompt/completion rows kept in the generations panel",
    )


class RLConfig(BaseModel):
    algorithm: Literal["grpo", "gspo", "cispo", "tis", "icepop"] = "grpo"

    # Optimization
    max_epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Core RL hyperparameters
    K: int = Field(default=8, description="Number of generations per prompt")
    eps_high: float = 0.28
    eps_low: float = 0.2
    kl: float = 0.0
    num_updates: int = 1

    # Dr. GRPO settings
    adv_normalize: bool = False
    length_normalize: bool = False

    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)

    # TIS
    C: float = Field(default=2.0, description="Importance-sampling clip for TIS")

    # IcePop
    icepop_a: float = 0.5
    icepop_b: float = 2.0


class MoEConfig(BaseModel):
    """Mixture-of-experts FFN (see llm.moe.MixtureOfExpert)."""

    num_experts: int = Field(description="Number of expert MLPs per MoE layer")
    top_k: int = Field(description="Experts activated per token (must be <= num_experts)")
    load_balancing_coef: float = Field(
        default=0.0,
        description="Scale for auxiliary load-balancing loss during training; 0 disables",
    )


class LLMConfig(BaseModel):
    num_layers: int = Field(description="Number of transformer layers")
    dim: int = Field(description="Model hidden dimension")
    num_head: int = Field(description="Number of attention heads")
    head_dim: int = Field(description="Dimension per attention head")
    max_seq: int = Field(description="Maximum sequence length")
    causal: bool = True
    pos_embed_type: Literal["rope", "absolute"] = "rope"
    ffn_type: Literal["mlp", "moe"] = Field(
        default="mlp",
        description="Feed-forward block: dense MLP or mixture-of-experts",
    )
    noise_scale: float = Field(
        default=0.01,
        description="Per-layer activation noise when forward(noise=True); simulates infer/train mismatch",
    )
    moe: MoEConfig | None = Field(
        default=None,
        description="Required when ffn_type is moe",
    )

    @model_validator(mode="after")
    def _validate_moe(self) -> "LLMConfig":
        if self.ffn_type == "moe":
            if self.moe is None:
                raise ValueError("llm_config.moe is required when ffn_type is 'moe'")
            if self.moe.top_k > self.moe.num_experts:
                raise ValueError(
                    f"moe.top_k ({self.moe.top_k}) must be <= moe.num_experts ({self.moe.num_experts})"
                )
        return self


class DataConfig(BaseModel):
    domain: str = Field(description="Task domain, e.g. dyck")
    train_size: int = Field(description="Number of training prompts")
    val_size: int = Field(description="Number of validation prompts")
    seed: int = Field(description="Random seed for data sampling")
    batch_size: int = 8
    max_sample_attempts: int = Field(
        description="Max get_task calls when collecting unique prompts",
    )
    oversample: bool = Field(
        default=False,
        description="If true, duplicate prompts when not enough uniques are found before timeout",
    )

    # Dyck task
    mean_prompt_length: int = 8
    prompt_length_min: int | None = None
    prompt_length_max: int | None = None
    max_completion_length: int = 16


class Config(BaseModel):
    rl_config: RLConfig
    llm_config: LLMConfig
    data_config: DataConfig
    visualize: VisualizeConfig = Field(default_factory=VisualizeConfig)
    device: str = "cpu"
