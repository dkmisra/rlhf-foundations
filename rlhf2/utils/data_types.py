from typing import Literal

from pydantic import BaseModel, Field


class InferenceConfig(BaseModel):
    """Sampling settings used during rollout / evaluation."""

    max_tokens: int = Field(default=40, description="Maximum tokens to generate per completion")
    temp: float = Field(default=1.0, description="Softmax temperature for sampling")


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

    # TIS
    C: float = Field(default=2.0, description="Importance-sampling clip for TIS")

    # IcePop
    icepop_a: float = 0.5
    icepop_b: float = 2.0


class ModelConfig(BaseModel):
    num_layers: int = Field(description="Number of transformer layers")
    dim: int = Field(description="Model hidden dimension")
    num_head: int = Field(description="Number of attention heads")
    head_dim: int = Field(description="Dimension per attention head")
    max_seq: int = Field(description="Maximum sequence length")
    causal: bool = True
    pos_embed_type: Literal["rope", "absolute"] = "rope"


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
    model_config: ModelConfig
    data_config: DataConfig
    device: str = "cpu"
