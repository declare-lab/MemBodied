import dataclasses
import logging
import threading
from typing import Any, Literal, Sequence
import einops
import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

class ReadContext(threading.local):
    def __init__(self):
        super().__init__()
        self.read_fn = None

_read_context = ReadContext()


# Helper functions replicated from openpi.models.gemma to ensure independence
def _apply_rope(x, *, positions, max_wavelength=10_000):
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return res.astype(x.dtype)


def _name(name, i):
    if i == 0:
        return name
    return f"{name}_{i}"


@at.typecheck
def posemb_sincos(pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float,
                  max_period: float) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period)**fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def anchor_update(z_prev, z_new, is_first):
    """The anchor's carry rule: where(is_first, new, old), broadcast over (N, D).

    Same shape as the S reset, one axis earlier -- the anchor is per-batch, not
    per-layer -- and its "init" is the current frame rather than a parameter.
    """
    return jnp.where(is_first[:, None, None], z_new, z_prev)


def resolve_first_frame_sequence(candidates, is_first):
    """Every step's first frame for a training window, as a (B, T, ...) pytree.

    Hold the window's opening frame and switch at each
    `is_first` to the frame supplied for the new episode.
    """
    def step(carry, t):
        current = jax.tree.map(lambda x: x[:, t], candidates)
        carry = jax.tree.map(
            lambda prev, new: jnp.where(
                is_first[:, t].reshape((-1,) + (1,) * (new.ndim - 1)), new, prev
            ),
            carry, current,
        )
        return carry, carry

    init = jax.tree.map(lambda x: x[:, 0], candidates)
    _, per_step = jax.lax.scan(step, init, jnp.arange(is_first.shape[1]))
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), per_step)


def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
class MemAttention(nn.Module):
    configs: Sequence[_gemma.Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache, layer_idx, memory_state=None, read_fn=None):
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = _gemma.lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = _gemma.lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, )),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = _gemma.lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim**-0.5
        k = _apply_rope(k, positions=positions)



        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(f"Attention mask shape mismatch: {attn_mask.shape} vs q={q.shape}, k={k.shape}")

        big_neg = -2.3819763e38
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = _gemma.lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)



        return out, (k, v)


@at.typecheck
class MemBlock(nn.Module):
    configs: Sequence[_gemma.Config]
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, kv_cache, memory_state, layer_idx, positions, attn_mask, deterministic=True):
        xs = _gemma.sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = MemAttention(configs=self.configs, name="attn")

        pre_attn = []
        for i, x in enumerate(xs):
            if x is not None:
                x = _gemma.RMSNorm(name=_name("pre_attention_norm", i))(x)
            pre_attn.append(x)

        pre_attn = _gemma.sharding.activation_sharding_constraint(pre_attn)
        
        # Read function passed through thread-local context
        read_fn = getattr(_read_context, "read_fn", None)
        
        # Contextual Integration (Pre-Attention Vector Injection)
        memory_gate_val = None
        if len(pre_attn) > 1 and pre_attn[1] is not None and read_fn is not None and memory_state is not None:
            # pre_attn[1] is suffix: [state, memory_token, actions]
            # memory_token is at index 1
            # h_mem = pre_attn[1][:, 1, :]
            # m_l, g_l = read_fn(layer_idx, h_mem) # m_l is projected vector, g_l is gate

            h_state_token = pre_attn[1][:, 0, :]
            m_l, g_l = read_fn(layer_idx, h_state_token) # m_l is projected vector, g_l is gate
            pre_attn[1] = pre_attn[1].at[:, 1, :].add(g_l * m_l)
            memory_gate_val = jnp.mean(g_l)

        # # Contextual Integration (Pre-Attention Vector Injection)
        # memory_gate_val = None
        # if len(pre_attn) > 1 and pre_attn[1] is not None and read_fn is not None and memory_state is not None:
        #     # pre_attn[1] is suffix: [state_token, actions...]
        #     # Inject memory directly into the state_token at index 0
        #     h_state_token = pre_attn[1][:, 0, :]
        #     m_l, g_l = read_fn(layer_idx, h_state_token) # m_l is projected vector, g_l is gate
        #     pre_attn[1] = pre_attn[1].at[:, 0, :].add(g_l * m_l)
        #     memory_gate_val = jnp.mean(g_l)

        post_attn, kv_cache = attn(
            pre_attn,
            positions,
            attn_mask,
            kv_cache,
            layer_idx=layer_idx,
            memory_state=memory_state,
            read_fn=read_fn
        )
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = _gemma.sharding.activation_sharding_constraint(post_attn)
        xs = jax.tree.map(lambda x, y: x + y, xs, post_attn)
        xs = _gemma.sharding.activation_sharding_constraint(xs)

        out = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x = _gemma.RMSNorm(name=_name("pre_ffw_norm", i))(x)
                x = _gemma.lora.FeedForward(
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)

        out = _gemma.sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = jax.tree.map(lambda x, y: x + y, xs, out)
        xs = _gemma.sharding.activation_sharding_constraint(xs)

        # # Collect state token representation at the output of this layer (input to next)
        # # Suffix is index 1. First token is state token.
        h_state = xs[1][:, 0, :] if xs[1] is not None else None

        # Collect memory token representation at the output of this layer (input to next).
        # Using memory token (position 1) as write key source instead of state token (position 0):
        # - Memory token already has the current-step memory injection applied pre-attention
        # - Memory token can attend to action tokens (state token cannot, due to causal mask)
        # - Decouples LoRA action generation gradients from the memory write gradient path
        # h_state = xs[1][:, 1, :] if xs[1] is not None else None
        

        return xs, (kv_cache, h_state, memory_gate_val)


@at.typecheck
class MemModule(nn.Module):
    configs: Sequence[_gemma.Config]
    embed_dtype: str
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    # Context fields to share with children in setup
    read_fn: Any = None

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = _gemma.Embedder(
            vocab_size=_gemma.PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        block_cls = nn.remat(
            MemBlock,
            prevent_cse=False,
            static_argnums=(7, ),  # 0=self, 7=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={
                "params": True,
                "dropout": True
            },
            in_axes=(0, 0, 0, nn.broadcast, nn.broadcast, nn.broadcast),  # 0=kv_cache, 1=memory_states, 2=layer_indices
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [_gemma.RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        *,
        kv_cache: _gemma.KVCache | None = None,
        memory_states: Any = None,
        deterministic: bool = True,
        read_fn: Any = None,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], _gemma.KVCache, at.Float[at.Array, "depth b d"] | None, at.Float[at.Array, "depth"] | None]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]

        depth = self.configs[0].depth
        batch_size = embedded[0].shape[0] if embedded[0] is not None else embedded[1].shape[0]

        if memory_states is None:
            # Create dummy states if not provided (shape: depth, B, H, r, r)
            memory_states = jnp.zeros((depth, batch_size, 50, 8, 8), dtype=embedded[0].dtype if embedded[0] is not None else embedded[1].dtype)

        # Setup scan inputs
        layer_indices = jnp.arange(depth)

        old_read_fn = getattr(_read_context, "read_fn", None)
        _read_context.read_fn = read_fn
        try:
            # Unpack outputs: layers scan returns final xs, and scanned (kv_cache, h_states, memory_gate_vals)
            embedded, (kv_cache, h_states, memory_gate_vals) = self.layers(
                embedded,
                kv_cache,
                memory_states,
                layer_indices,
                positions,
                mask,
                deterministic,
            )
        finally:
            _read_context.read_fn = old_read_fn

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        final_out = [f(e) if e is not None else e for f, e in zip(self.final_norms, embedded, strict=True)]
        return final_out, kv_cache, h_states, memory_gate_vals

    def init(self, rngs=None, *args, **kwargs):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
        )


@dataclasses.dataclass(frozen=True)
class Pi0MemConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Memory hyperparameters
    memory_rank: int = 8
    memory_alpha: float = 16.0
    beta_bias_init: float = -1.5
    alpha_bias_init: float = 2.0
    sequence_len: int | None = 10  # Training sequence length, None/<=1 triggers single step
    sequence_stride: int = 50  # Training sequence step stride (e.g. 50 matches pi0_step)
    truncate_gradients: bool = True  # Truncates gradients across temporal steps (Method B)
    
    memory_type: Literal["base", "lstm_cell"] = "base"
    memory_gamma: float = 0.1       # mixing coefficient
    memory_value_type: Literal["vision_action", "vision", "action"] = "vision_action"
    memory_dropout: float = 0.0         # dropout on the memory token vector
    mem_init: bool = True

    # --- anchor memory -----------------------------------------------------
    # A fixed vision encoding of the episode's FIRST frame. The current frame
    # cross-attends into it and the result conditions the action expert. It is
    # constant within an episode, so it never enters the TBPTT graph. Kept
    # independent of the associative memory so the two ablate separately.
    use_anchor_memory: bool = False

    # --- first-frame conditioning ------------------------------------------
    # The episode's first frame appended to the prefix as one more camera, at
    # full resolution. This is the paper's first-frame ablation without
    # pooling, gating, cross-attention, or new parameters.
    use_first_frame: bool = False

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    def __post_init__(self) -> None:
        if self.memory_type not in ("base", "lstm_cell"):
            raise ValueError(f"Unsupported memory_type: {self.memory_type!r}")
        if self.memory_value_type not in ("vision_action", "vision", "action"):
            raise ValueError(f"Unsupported memory_value_type: {self.memory_value_type!r}")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Mem":
        return Pi0Mem(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True

        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class AnchorConcatKernelInit:
    action_expert_width: int
    def __call__(self, key, shape, dtype=jnp.float32):
        expected_shape = (3 * self.action_expert_width, self.action_expert_width)
        if shape != expected_shape:
            raise ValueError(
                f"anchor concat kernel expected shape {expected_shape}, got {shape}"
            )
        pretrained_slice = nn.initializers.lecun_normal()(
            key, (2 * self.action_expert_width, self.action_expert_width), dtype
        )
        anchor_slice = jnp.zeros(
            (self.action_expert_width, self.action_expert_width), dtype=dtype
        )
        return jnp.concatenate([pretrained_slice, anchor_slice], axis=0)


class Pi0Mem(_model.BaseModel):
    def __init__(self, config: Pi0MemConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Customized memory transformer
        llm = nnx_bridge.ToNNX(
            MemModule(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            ))
        llm.lazy_init(rngs=rngs, method="init")
        
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            ))
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        action_expert_width = action_expert_config.width
        if config.use_anchor_memory:
            # This is one literal 3*width affine map, as opposed to a parallel
            # side projection.  Its first 2*width rows use exactly NNX Linear's
            # normal LeCun initialization and its new third slice is zero.  A
            # pretrained 2*width checkpoint is widened the same way by the
            # checkpoint loader below, so both fresh init and checkpoint load
            # begin as the exact pre-anchor function.
            self.action_time_mlp_in = nnx.Linear(
                3 * action_expert_width,
                action_expert_width,
                rngs=rngs,
                kernel_init=AnchorConcatKernelInit(action_expert_width),
            )
        else:
            self.action_time_mlp_in = nnx.Linear(
                2 * action_expert_width, action_expert_width, rngs=rngs
            )
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Layer-specific Read/Write Projection layers
        depth = action_expert_config.depth
        
        if config.memory_value_type == "vision":
            value_dim = paligemma_config.width
            self.vision_dim = paligemma_config.width
        elif config.memory_value_type == "vision_action":
            value_dim = paligemma_config.width + config.action_dim
            self.vision_dim = paligemma_config.width
        else:
            value_dim = config.action_dim
            self.vision_dim = None

        in_dim = action_expert_config.width
        gate_in_dim = action_expert_config.width + value_dim

        self.read_proj_q = [nnx.Linear(in_dim, config.memory_rank, rngs=rngs) for _ in range(depth)]

        # self.read_proj_memory = [nnx.Linear(config.memory_rank, action_expert_config.width, use_bias=True, rngs=rngs) for _ in range(depth)]
        # self.memory_gate = [nnx.Linear(action_expert_config.width, 1, use_bias=True, rngs=rngs) for _ in range(depth)]
        # Changing INIT to 0s
        self.read_proj_memory = [nnx.Linear(config.memory_rank, action_expert_config.width, use_bias=True, rngs=rngs, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros) for _ in range(depth)]
        self.memory_gate = [nnx.Linear(action_expert_config.width, 1, use_bias=True, rngs=rngs, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros) for _ in range(depth)]


        # Initialize memory token
        if self.config.mem_init:
            self.memory_token_init = nnx.Param(
                jax.random.normal(rngs.params() if hasattr(rngs, "params") else rngs(), (1, 1, action_expert_config.width)) * 0.02
            )
        # else:
        #     # Initialize memory token (Frozen Zero)
        #     self.memory_token_init = jnp.zeros((1, 1, action_expert_config.width))

        self.write_proj_k = [nnx.Linear(in_dim, config.memory_rank, rngs=rngs) for _ in range(depth)]
        self.write_proj_v = [nnx.Linear(value_dim, config.memory_rank, rngs=rngs) for _ in range(depth)]
        # self.write_proj_beta = [nnx.Linear(gate_in_dim, 1, rngs=rngs) for _ in range(depth)]
        self.write_proj_beta = [nnx.Linear(gate_in_dim, config.memory_rank, rngs=rngs) for _ in range(depth)]
        self.write_proj_alpha = [nnx.Linear(gate_in_dim, config.memory_rank, rngs=rngs) for _ in range(depth)]

        # Trainable initial memory state parameter
        self.S_init_param = nnx.Param(
            jax.random.normal(rngs.params() if hasattr(rngs, "params") else rngs(), (depth, 1, config.memory_rank, config.memory_rank)) * 0.02
        )

        needs_vision_encoding = (
            config.memory_value_type in ("vision", "vision_action")
            or config.use_anchor_memory
        )
        if needs_vision_encoding:
            D = paligemma_config.width
            self.pooling_proj_q = nnx.Linear(action_expert_config.width, D, rngs=rngs)
            self.pooling_proj_k = nnx.Linear(D, D, rngs=rngs)
            self.pooling_proj_v = nnx.Linear(D, D, rngs=rngs)
            
        if config.memory_type == "lstm_cell":
            self.lstm_forget = [
                nnx.Linear(action_expert_config.width, config.memory_rank, rngs=rngs)
                for _ in range(depth)
            ]

        # --- anchor memory -------------------------------------------------
        # The current frame's encoding queries the episode's first-frame anchor
        # through the paper-fixed rank-64 cross-attention pathway.
        self.vision_width = paligemma_config.width
        if config.use_anchor_memory:
            vwidth = self.vision_width
            r = 64
            self.anchor_q = nnx.Linear(vwidth, r, rngs=rngs)
            self.anchor_k = nnx.Linear(vwidth, r, rngs=rngs)
            self.anchor_v = nnx.Linear(vwidth, r, rngs=rngs)

            self.anchor_o = nnx.Linear(r, vwidth, rngs=rngs)
            self.anchor_proj = nnx.Linear(vwidth, action_expert_config.width, rngs=rngs)

        # Store action expert attention dims for use in get_read_fn reshape
        self._action_expert_num_heads = action_expert_config.num_heads

    def image_tokens(self, obs: _model.Observation) -> list:
        """SigLIP tokens per camera. Split out so the anchor can reuse the very
        tokens the prefix is built from rather than encoding the frame twice."""
        return [self.PaliGemma.img(obs.images[name], train=False)[0] for name in obs.images]

    def episode_anchor_image_tokens(
        self,
        obs: _model.Observation,
        *,
        current_image_tokens: list | None = None,
    ) -> list:
        """Encode the true episode-first images supplied by sequence training.

        Inference starts on the episode's first observation, so it can reuse the
        current prefix tokens. A random training window can begin mid-episode;
        in that case the dataset supplies the actual first frame separately.
        """
        if obs.episode_anchor_images is None:
            return current_image_tokens if current_image_tokens is not None else self.image_tokens(obs)
        if obs.episode_anchor_image_masks is None:
            raise ValueError("episode anchor images are missing their image masks")
        anchor_obs = obs.replace(
            images=obs.episode_anchor_images,
            image_masks=obs.episode_anchor_image_masks,
            episode_anchor_images=None,
            episode_anchor_image_masks=None,
        )
        return self.image_tokens(anchor_obs)

    def first_frame_camera(self, images: dict) -> str:
        """The camera whose first frame goes into the prefix.

        One camera, not all three: the prefix grows by a single 256-token block
        (768 -> 1024 here). The first entry is the external view in every config
        on this arm, and a wrist camera at t=0 says little about the scene.
        """
        return next(iter(images))

    def episode_first_frame(self, obs: _model.Observation) -> tuple[dict, dict]:
        """The episode's first-frame image and mask, as one-entry dicts.

        Prefers the literal first frame the dataset supplies, since a training
        window can open mid-episode; falls back to the current frame, which is
        exactly what inference sees on the first observation after
        `Policy.reset()`. The mask follows the frame it belongs to, so at
        `is_first` it is the current frame's own mask.

        `stop_gradient` is explicit: the stored frame is fixed context for the
        episode and must not enter the TBPTT graph. The encoder still trains on
        it the same way it trains on any other prefix camera.
        """
        if obs.episode_anchor_images is not None:
            if obs.episode_anchor_image_masks is None:
                raise ValueError("episode anchor images are missing their image masks")
            images, masks = obs.episode_anchor_images, obs.episode_anchor_image_masks
        else:
            images, masks = obs.images, obs.image_masks
        name = self.first_frame_camera(images)
        return {name: jax.lax.stop_gradient(images[name])}, {name: masks[name]}

    def first_frame_state(self, observation: _model.Observation) -> tuple[dict, dict]:
        """The first-frame images `Policy` holds for the rest of an episode.

        Preprocessing is deterministic at `train=False`, so what this returns is
        bit-identical to the images the current-frame prefix path builds from
        the same observation.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        return self.episode_first_frame(observation)

    def extract_anchor(self, image_tokens: list) -> at.Float[at.Array, "b n d"]:
        """Pool a frame's image tokens into the anchor encoding.

        Stopped-gradient, matching `get_vision_encoding`: the anchor is a fixed
        description of where the episode started, not something the memory path
        is allowed to reshape the vision encoder to produce.
        """
        pooled = []
        for cam_tokens in image_tokens:
            stopped_tokens = jax.lax.stop_gradient(cam_tokens)
            pooled.append(self._pool_grid(stopped_tokens))
        return jnp.concatenate(pooled, axis=1)

    def _pool_grid(self, tokens: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b m d"]:
        """Average-pool one camera's square token grid to the paper-fixed 4x4 grid."""
        g = 4
        num = tokens.shape[1]
        side = round(num ** 0.5)
        if side * side != num or side % g != 0:
            raise ValueError(
                f"grid anchors need a square token grid divisible by {g}, "
                f"got {num} tokens ({side}x{side})"
            )
        block = side // g
        # Flat index is row*side + col, so (gh ph gw pw) is the g-by-g blocking.
        grid = einops.rearrange(
            tokens, "b (gh ph gw pw) d -> b gh gw (ph pw) d",
            gh=g, ph=block, gw=g, pw=block,
        )
        return einops.rearrange(jnp.mean(grid, axis=3), "b gh gw d -> b (gh gw) d")

    def read_anchor(
        self,
        z_cur: at.Float[at.Array, "b d"],
        z_anchor: at.Float[at.Array, "b n d"],
    ) -> tuple[at.Float[at.Array, "b d"], at.Array]:
        """Cross-attend the current frame's encoding into the anchor.

        Returns the fused encoding and the magnitude of the anchor's own
        contribution -- the concat site's counterpart to the prefix site's gate,
        and what gets logged for it.
        """
        r = 64
        q = self.anchor_q(z_cur)[:, None, :]        # (B, 1, r)
        k = self.anchor_k(z_anchor)                 # (B, N, r)
        v = self.anchor_v(z_anchor)                 # (B, N, r)
        logits = jnp.einsum("bqr,bnr->bqn", q, k) / jnp.sqrt(float(r))
        read = self.anchor_o(jnp.einsum("bqn,bnr->bqr", jax.nn.softmax(logits, axis=-1), v))[:, 0]
        magnitude = jnp.mean(jnp.linalg.norm(read.astype(jnp.float32), axis=-1))
        return read + z_cur, magnitude

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation, image_tokens: list | None = None,
        first_frame: tuple[dict, dict] | None = None,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        if image_tokens is None:
            image_tokens = self.image_tokens(obs)
        for name, cam_tokens in zip(obs.images, image_tokens, strict=True):
            tokens.append(cam_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=cam_tokens.shape[1]))
            ar_mask += [False] * cam_tokens.shape[1]

        # The episode's first frame, as one more camera: same SigLIP encoder,
        # after the current cameras, before the text, and ar_mask False like
        # every other image token so the prefix stays one bidirectional block.
        if self.config.use_first_frame:
            if first_frame is None:
                first_frame = self.episode_first_frame(obs)
            first_frame_images, first_frame_masks = first_frame
            for name, image in first_frame_images.items():
                first_frame_tokens, _ = self.PaliGemma.img(image, train=False)
                tokens.append(first_frame_tokens)
                input_mask.append(einops.repeat(
                    first_frame_masks[name], "b -> b s", s=first_frame_tokens.shape[1]))
                ar_mask += [False] * first_frame_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def get_vision_encoding(
        self, obs: _model.Observation, image_tokens: list | None = None
    ) -> at.Float[at.Array, "b d"]:
        # `image_tokens` lets callers hand in the SigLIP output the prefix already
        # computed instead of paying for a second encode of the same frame.
        if image_tokens is None:
            image_tokens = self.image_tokens(obs)
        tokens = [jax.lax.stop_gradient(t) for t in image_tokens]

        pooled_list = []
        for cam_tokens in tokens:
            state_feat = self.state_proj(obs.state)
            q = self.pooling_proj_q(state_feat)
            q = jnp.expand_dims(q, axis=1)
            k = self.pooling_proj_k(cam_tokens)
            v = self.pooling_proj_v(cam_tokens)
            D = q.shape[-1]
            logits = jnp.einsum("bqd,bnd->bqn", q, k) / jnp.sqrt(D)
            attn = jax.nn.softmax(logits, axis=-1)
            pooled_tokens = jnp.einsum("bqn,bnd->bqd", attn, v)
            pooled_list.append(jnp.squeeze(pooled_tokens, axis=1))
        vision_encoding = jnp.mean(jnp.stack(pooled_list, axis=0), axis=0)
            
        return vision_encoding

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"],
        anchor_cond: at.Float[at.Array, "b d"] | None = None,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        ar_mask += [True]
        
        # Comment out below block to remove Memory token insertion.
        # Insert memory token
        B = obs.state.shape[0]
        if self.config.mem_init:
            token_val = self.memory_token_init.value
        else:
            token_val = jnp.zeros((1, 1, state_token.shape[-1]))
        memory_token = jnp.broadcast_to(token_val, (B, 1, state_token.shape[-1]))
        tokens.append(memory_token)
        input_mask.append(jnp.ones((B, 1), dtype=jnp.bool_))
        ar_mask += [True]

        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_inputs = [action_tokens, time_tokens]
        if self.config.use_anchor_memory:
            if anchor_cond is None:
                # Useful for the zero-init identity check and fail-safe direct
                # calls: the real 3*width map still receives its third slice.
                z_b = jnp.zeros_like(action_tokens)
            else:
                z_b = einops.repeat(
                    self.anchor_proj(anchor_cond), "b d -> b s d", s=self.action_horizon
                )
                z_b = z_b.astype(action_tokens.dtype)
            action_time_inputs.append(z_b)
        action_time_tokens = self.action_time_mlp_in(
            jnp.concatenate(action_time_inputs, axis=-1)
        )
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def get_read_fn(self, memory_states, train: bool = False, rng: at.KeyArrayLike | None = None):
        if isinstance(memory_states, (tuple, list)):
            S_state = memory_states[0]
        else:
            S_state = memory_states

        def read_fn(layer_idx, h_state):
            # h_state shape: (B, D)
            q = jnp.tanh(jax.lax.switch(layer_idx, self.read_proj_q, h_state))
            q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
            r = jnp.einsum("bhij,bj->bhi", S_state[layer_idx], q)

            m_l = jax.lax.switch(layer_idx, self.read_proj_memory, r)  # shape: (B, H, width)
            m_l = jnp.mean(m_l, axis=1)  # shape: (B, width)
            
            g_l = jax.lax.switch(layer_idx, self.memory_gate, h_state) # shape: (B, 1)
            g_l = jax.nn.sigmoid(g_l)
            
            scale = self.config.memory_alpha / self.config.memory_rank
            m_l = (m_l * scale).astype(h_state.dtype)

            if train and self.config.memory_dropout > 0.0 and rng is not None:
                layer_rng = jax.random.fold_in(rng, layer_idx)
                keep_prob = 1.0 - self.config.memory_dropout
                mask = jax.random.bernoulli(layer_rng, p=keep_prob, shape=m_l.shape)

                m_l = jax.lax.select(mask, m_l / keep_prob, jnp.zeros_like(m_l))

            return m_l, g_l
        return read_fn

    def write_memory(
        self,
        h_states: at.Float[at.Array, "depth b d"],
        value: Any,
        memory_states: Any,
        *,
        return_metrics: bool = False
    ) -> Any:
        """Write value into the episodic memory state.

        Args:
            h_states: Hidden states from the action expert, shape (depth, B, D).
            value: The value vector to write. When memory_value_type=="vision" this is the
                   concatenated [vision_encoding, actions] of shape (B, H, vision_dim + action_dim).
                   When memory_value_type=="action" this is the action chunk (B, H, action_dim).
            memory_states: Current memory state.
        """
        depth = len(self.write_proj_k)

        if isinstance(memory_states, (tuple, list)):
            S_state, C_state = memory_states
        else:
            S_state = memory_states
            C_state = None

        new_S_states = []
        new_C_states = []
        betas = []
        alphas = []

        for l in range(depth):
            h_state = h_states[l]  # shape: (B, D)

            S_prev = S_state[l]

            k = jnp.tanh(self.write_proj_k[l](h_state))
            k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-12)

            v = self.write_proj_v[l](value)

            write_h = value.shape[1]
            h_state_expanded = einops.repeat(h_state, "b d -> b h d", h=write_h)
            gate_input = jnp.concatenate([h_state_expanded, value], axis=-1)
            beta = jax.nn.sigmoid(self.write_proj_beta[l](gate_input) + self.config.beta_bias_init)
            alpha = jax.nn.sigmoid(self.write_proj_alpha[l](gate_input) + self.config.alpha_bias_init)

            betas.append(jnp.mean(beta))
            alphas.append(jnp.mean(alpha))

            S_prev_alpha_k = jnp.einsum("bhij,bj->bhi", S_prev, k) * alpha
            diff = v - S_prev_alpha_k
            update_term = jnp.einsum("bhi,bj->bhij", diff, k)
            S_next = alpha[..., None] * S_prev + beta[..., None] * update_term

            if self.config.memory_type == "lstm_cell":
                f = jax.nn.sigmoid(self.lstm_forget[l](h_state))
                S_tilde = jnp.mean(S_next, axis=1)

                C_prev = C_state[l]
                C_next = f[..., None] * C_prev + (1.0 - f)[..., None] * S_tilde
                new_C_states.append(C_next)

                gamma = self.config.memory_gamma
                new_S_states.append((1.0 - gamma) * S_next + gamma * C_next[:, None, :, :])
            else:
                new_S_states.append(S_next)

        S_next_stacked = jnp.stack(new_S_states, axis=0)

        if self.config.memory_type == "lstm_cell":
            C_next_stacked = jnp.stack(new_C_states, axis=0)
            carry_next = (S_next_stacked, C_next_stacked)
        else:
            carry_next = S_next_stacked

        if return_metrics:
            metrics = {
                "beta_mean": jnp.mean(jnp.stack(betas)) if len(betas) > 0 else jnp.zeros(()),
                "alpha_mean": jnp.mean(jnp.stack(alphas)) if len(alphas) > 0 else jnp.zeros(()),
            }
            return carry_next, metrics
        return carry_next

    @override
    def compute_loss(self,
                     rng: at.KeyArrayLike,
                     observation: _model.Observation,
                     actions: _model.Actions,
                     *,
                     train: bool = False,
                     return_metrics: bool = False) -> Any:
        preprocess_rng, noise_rng, time_rng, dropout_rng = jax.random.split(rng, 4)

        # Check if actions has sequence dimension (Batch, Time, H, action_dim)
        if actions.ndim == 4:
            # Sequence training with causally delayed writes and optional gradient truncation.
            batch_size = actions.shape[0]
            seq_len = actions.shape[1]
            depth = len(self.read_proj_q)

            # Initialize memory carry state based on memory_type
            state_horizon = 1 if self.config.memory_value_type in ("vision", "vision_action") else self.config.action_horizon
            if self.config.memory_type == "lstm_cell":
                S_init = jnp.broadcast_to(self.S_init_param.value[:, None], (depth, batch_size, state_horizon, self.config.memory_rank, self.config.memory_rank))
                C_init = jnp.zeros((depth, batch_size, self.config.memory_rank, self.config.memory_rank), dtype=jnp.float32)
                carry_init = (S_init, C_init)
            else:
                S_init = jnp.broadcast_to(self.S_init_param.value[:, None], (depth, batch_size, state_horizon, self.config.memory_rank, self.config.memory_rank))
                carry_init = S_init

            # 1. Parallel Preprocessing & Prefix precomputation
            preprocess_rng_seq = jax.random.split(preprocess_rng, seq_len)
            
            # vmap observation preprocessing across Time dimension (axis 1)
            obs_in_axes = jax.tree.map(lambda x: 1 if isinstance(x, jax.Array) or hasattr(x, "shape") else None, observation)
            preprocess_fn = jax.vmap(
                lambda r, o: _model.preprocess_observation(r, o, train=train),
                in_axes=(0, obs_in_axes),
                out_axes=obs_in_axes
            )
            observation = preprocess_fn(preprocess_rng_seq, observation)
            
            # Flatten Batch and Time dimensions to run SigLIP and LLM prefix pass in parallel
            def flatten_bt(x):
                if isinstance(x, jax.Array) or hasattr(x, "shape"):
                    return einops.rearrange(x, "b t ... -> (b t) ...")
                return x
            
            obs_flat = jax.tree.map(flatten_bt, observation)
            
            # One SigLIP pass over (batch, time). The prefix, the delta path's
            # vision value and the anchor all read from these same tokens rather
            # than each re-encoding the frame.
            image_tokens_flat = self.image_tokens(obs_flat)
            anchor_concat = self.config.use_anchor_memory

            if self.config.memory_value_type in ("vision", "vision_action") or anchor_concat:
                vision_enc_flat = self.get_vision_encoding(obs_flat, image_tokens=image_tokens_flat)
                vision_enc = einops.rearrange(vision_enc_flat, "(b t) d -> b t d", b=batch_size, t=seq_len)
            else:
                vision_enc = None

            if self.config.memory_value_type in ("vision", "vision_action"):
                next_vision_enc = jnp.concatenate([vision_enc[:, 1:], vision_enc[:, -1:]], axis=1)
            else:
                next_vision_enc = None

            if getattr(observation, "is_first_step", None) is not None:
                is_first_seq = observation.is_first_step
            else:
                is_first_seq = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)

            # --- anchor memory -------------------------------------------------
            # Extraction is parameter-free and stop-gradiented, so every candidate
            # is a constant: whichever one the carry holds, the anchor never
            # enters the TBPTT graph.
            anchor_candidates = None
            if self.config.use_anchor_memory:
                anchor_image_tokens_flat = self.episode_anchor_image_tokens(
                    obs_flat, current_image_tokens=image_tokens_flat
                )
                anchor_candidates = einops.rearrange(
                    self.extract_anchor(anchor_image_tokens_flat), "(b t) n d -> b t n d",
                    b=batch_size, t=seq_len,
                )

            # --- first-frame conditioning ------------------------------------
            # Resolved for the whole window up front, because the prefix KV cache
            # is precomputed in parallel over (batch, time) and so needs every
            # step's frame before the scan starts. Same where(is_first, current,
            # previous) rule the scan applies to S, and stop_gradient in
            # `episode_first_frame` keeps it out of the TBPTT graph.
            first_frame_flat = None
            if self.config.use_first_frame:
                first_frame_candidates = jax.tree.map(
                    lambda x: einops.rearrange(
                        x, "(b t) ... -> b t ...", b=batch_size, t=seq_len),
                    self.episode_first_frame(obs_flat),
                )
                first_frame_flat = jax.tree.map(
                    lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
                    resolve_first_frame_sequence(first_frame_candidates, is_first_seq),
                )

            # Compute parallel prefix tokens and mask
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
                obs_flat, image_tokens=image_tokens_flat, first_frame=first_frame_flat)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            
            # Compute parallel prefix KV cache
            _, prefix_kv_cache_flat, _, _ = self.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=positions,
            )
            
            # Reshape KV cache back to (layers, Batch, Time, SeqLen, Heads, Dim)
            prefix_kv_cache = jax.tree.map(
                lambda x: einops.rearrange(x, "l (b t) s h d -> l b t s h d", b=batch_size, t=seq_len),
                prefix_kv_cache_flat
            )
            
            # Reshape prefix_mask back to (Batch, Time, SeqLen)
            prefix_mask_reshaped = einops.rearrange(prefix_mask, "(b t) s -> b t s", b=batch_size, t=seq_len)

            # z_anchor rides in the scan carry beside S for both fusion sites.
            # Prefix fusion consumed the resolved per-step values above, while
            # concat reads the live carried value below; keeping the same carry
            # rule for both makes episode-boundary semantics explicit.
            anchor_carry_init = anchor_candidates[:, 0] if anchor_candidates is not None else None

            def step_fn(carry, t):
                mem_carry, z_anchor = carry
                # Slice observation and actions at step t
                obs_t = jax.tree.map(lambda x: x[:, t] if isinstance(x, jax.Array) or hasattr(x, "shape") else x, observation)
                act_t = actions[:, t]

                # --- NEW CODE: Reset memory state on episode boundaries ---
                if getattr(observation, "is_first_step", None) is not None:
                    reset_mask = observation.is_first_step[:, t]  # shape: (Batch,)
                    
                    def reset_where_true(c, c_init):
                        # c shape: [depth, Batch, ...]
                        shape = [1] * c.ndim
                        if c.ndim >= 2:
                            shape[1] = reset_mask.shape[0]  # Match the Batch dimension
                        mask_expanded = reset_mask.reshape(shape)
                        return jnp.where(mask_expanded, c_init, c)
                        
                    mem_carry = jax.tree.map(reset_where_true, mem_carry, carry_init)
                    if z_anchor is not None:
                        # Same reset, one step further: the anchor is replaced by
                        # this frame's own encoding and then held for the episode.
                        z_anchor = anchor_update(z_anchor, anchor_candidates[:, t], reset_mask)
                # --------------------------------------------------------

                # Split RNG for the step
                step_rng = jax.random.fold_in(rng, t)
                
                # Slice precomputed prefix KV cache and prefix mask for step t
                kv_cache_t = jax.tree.map(lambda x: x[:, :, t], prefix_kv_cache)
                prefix_mask_t = prefix_mask_reshaped[:, t]

                val_t = next_vision_enc[:, t] if next_vision_enc is not None else None

                loss_t, next_carry, metrics_t = self._forward_step(
                    step_rng,
                    obs_t,
                    act_t,
                    mem_carry,
                    kv_cache=kv_cache_t,
                    prefix_mask=prefix_mask_t,
                    value_input=val_t,
                    anchor_state=z_anchor,
                    vision_enc=vision_enc[:, t] if vision_enc is not None else None,
                    train=train,
                    return_metrics=True
                )

                if observation.loss_mask is not None:
                    loss_t = loss_t * observation.loss_mask[:, t][:, None]

                # Detach state from backward graph to stop outer-loop sequential gradients if truncate_gradients is enabled
                next_carry = jax.lax.cond(
                    train & self.config.truncate_gradients,
                    lambda: jax.tree.map(jax.lax.stop_gradient, next_carry),
                    lambda: next_carry
                )

                # # Add relative noise to carry for regularization (prevents temporal memorization)
                # if train and self.config.carry_noise_std > 0.0:
                #     noise_rng = jax.random.fold_in(rng, t + seq_len)
                #     def add_relative_noise(x):
                #         carry_rms = jnp.sqrt(jnp.mean(jnp.square(x)))
                #         noise_scale = self.config.carry_noise_std * jnp.maximum(carry_rms, 1e-6)
                #         return x + jax.random.normal(noise_rng, x.shape, dtype=x.dtype) * noise_scale
                #     next_carry = jax.tree.map(add_relative_noise, next_carry)
                
                # if train and self.config.carry_dropout > 0.0:
                #     dropout_rng = jax.random.fold_in(rng, t + seq_len)
                #     keep_prob = 1.0 - self.config.carry_dropout
                #     def apply_carry_dropout(x):
                #         mask = jax.random.bernoulli(dropout_rng, p=keep_prob, shape=x.shape)
                #         return jnp.where(mask, x / keep_prob, jnp.zeros_like(x))  # inverted dropout
                #     next_carry = jax.tree.map(apply_carry_dropout, next_carry)

                #     # Log actual zero fraction as a metric (should track ~carry_dropout config value)
                #     S = next_carry[0] if isinstance(next_carry, tuple) else next_carry
                #     metrics_t["carry_dropout_frac"] = jnp.mean((S == 0.0).astype(jnp.float32))


                return (next_carry, z_anchor), (loss_t, metrics_t)

            # Scan over sequence steps
            _, (loss_seq, metrics_seq) = jax.lax.scan(
                step_fn, (carry_init, anchor_carry_init), jnp.arange(seq_len))
            # Transpose sequence loss back to match batch shape
            # jax.lax.scan returns shape (seq_len, B, H)
            loss_seq = jnp.transpose(loss_seq, (1, 0, 2))

            if observation.loss_mask is not None:
                # Dynamically scale the loss so the downstream .mean() reflects only valid frames
                valid_frames = jnp.sum(observation.loss_mask, axis=1)  # shape: (B,)
                scale = seq_len / jnp.maximum(valid_frames, 1.0)
                loss_seq = loss_seq * scale[:, None, None]
                mean_valid_frames = jnp.mean(valid_frames)
            else:
                mean_valid_frames = jnp.array(seq_len, dtype=jnp.float32)

            # Average metrics across sequence steps
            avg_metrics = jax.tree.map(jnp.mean, metrics_seq)
            avg_metrics["num_valid_frames"] = mean_valid_frames
            if return_metrics:
                return loss_seq, avg_metrics
            return loss_seq

        else:
            # Single-step training / validation
            # Initialize carry state if not passed
            observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
            depth = len(self.read_proj_q)
            batch_size = actions.shape[0]

            state_horizon = 1 if self.config.memory_value_type in ("vision", "vision_action") else self.config.action_horizon
            if self.config.memory_type == "lstm_cell":
                S_init = jnp.broadcast_to(self.S_init_param.value[:, None], (depth, batch_size, state_horizon, self.config.memory_rank, self.config.memory_rank))
                C_init = jnp.zeros((depth, batch_size, self.config.memory_rank, self.config.memory_rank), dtype=jnp.float32)
                carry_init = (S_init, C_init)
            else:
                carry_init = jnp.broadcast_to(self.S_init_param.value[:, None], (depth, batch_size, state_horizon, self.config.memory_rank, self.config.memory_rank))
                
            image_tokens = self.image_tokens(observation)
            anchor_concat = self.config.use_anchor_memory

            uses_vision_value = self.config.memory_value_type in ("vision", "vision_action")
            if uses_vision_value or anchor_concat:
                vision_enc = self.get_vision_encoding(observation, image_tokens=image_tokens)
            else:
                vision_enc = None
            val_input = vision_enc if uses_vision_value else None

            # A single step is by definition the start of its own episode, so
            # this frame is the anchor.
            anchor_state = (
                self.extract_anchor(
                    self.episode_anchor_image_tokens(
                        observation, current_image_tokens=image_tokens
                    )
                )
                if self.config.use_anchor_memory
                else None
            )

            loss, _, metrics = self._forward_step(
                rng, observation, actions, carry_init, value_input=val_input,
                anchor_state=anchor_state, vision_enc=vision_enc, image_tokens=image_tokens,
                train=train, return_metrics=True
            )
            if return_metrics:
                return loss, metrics
            return loss

    def _forward_step(
        self,
        rng,
        observation,
        actions,
        carry_prev,
        *,
        kv_cache: _gemma.KVCache | None = None,
        prefix_mask: at.Bool[at.Array, "b s"] | None = None,
        value_input: Any = None,
        anchor_state: Any = None,
        vision_enc: Any = None,
        image_tokens: list | None = None,
        train: bool = False,
        return_metrics: bool = False
    ):
        preprocess_rng, noise_rng, time_rng, dropout_rng = jax.random.split(rng, 4)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # --- anchor memory, concat site ------------------------------------
        # The current frame queries the episode anchor and the result conditions
        # the action expert. `vision_enc` is the caller's already-computed
        # encoding of this frame, so the sequence path does not re-run SigLIP
        # inside the scan.
        anchor_cond = None
        anchor_attn_mag = jnp.zeros(())
        if anchor_state is not None:
            z_cur = vision_enc if vision_enc is not None else self.get_vision_encoding(observation)
            anchor_cond, anchor_attn_mag = self.read_anchor(z_cur, anchor_state)

        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
            observation, x_t, time, anchor_cond=anchor_cond)

        if kv_cache is not None:
            assert prefix_mask is not None
            suffix_len = suffix_tokens.shape[1]
            prefix_attn_mask_rep = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
            
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            full_attn_mask = jnp.concatenate([prefix_attn_mask_rep, suffix_attn_mask], axis=-1)
            
            positions_suffix = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _, h_states, memory_gate_vals = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_suffix,
                kv_cache=kv_cache,
                memory_states=carry_prev,
                read_fn=self.get_read_fn(carry_prev, train=train, rng=dropout_rng),
            )
        else:
            if image_tokens is None:
                image_tokens = self.image_tokens(observation)
            prefix_tokens, prefix_mask_val, prefix_ar_mask = self.embed_prefix(
                observation, image_tokens=image_tokens)
            input_mask = jnp.concatenate([prefix_mask_val, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1

            (prefix_out, suffix_out), _, h_states, memory_gate_vals = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                memory_states=carry_prev,
                read_fn=self.get_read_fn(carry_prev, train=train, rng=dropout_rng),
            )

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # Update memory state using the next-frame value.
        if self.config.memory_value_type == "vision":
            if value_input is not None:
                val = value_input
            else:
                B, _, _ = actions.shape
                val = jnp.zeros((B, self.vision_dim), dtype=actions.dtype)
            val = jnp.expand_dims(val, axis=1)  # shape: (B, 1, vision_dim)
        elif self.config.memory_value_type == "vision_action":
            B, H, D_a = actions.shape
            pooled_actions = jnp.sum(actions, axis=1)
                
            if value_input is not None:
                val = jnp.concatenate([value_input, pooled_actions], axis=-1)  # shape: (B, vision_dim + action_dim)
            else:
                dummy_vision = jnp.zeros((B, self.vision_dim), dtype=actions.dtype)
                val = jnp.concatenate([dummy_vision, pooled_actions], axis=-1)
            val = jnp.expand_dims(val, axis=1)  # shape: (B, 1, vision_dim + action_dim)
        else:
            val = actions

        carry_next, write_metrics = self.write_memory(h_states, val, carry_prev, return_metrics=True)

        # Calculate Frobenius norm of S_prev
        S_prev = carry_prev[0] if isinstance(carry_prev, tuple) else carry_prev
        S_norm = jnp.mean(jnp.sqrt(jnp.sum(jnp.square(S_prev), axis=(-2, -1))))

        S_next = carry_next[0] if isinstance(carry_next, tuple) else carry_next
        S_change = S_next - S_prev
        S_change_norm = jnp.mean(jnp.sqrt(jnp.sum(jnp.square(S_change), axis=(-2, -1))))

        # Calculate memory gate mean
        if memory_gate_vals is not None:
            memory_gate_mean = jnp.mean(memory_gate_vals)
        else:
            memory_gate_mean = jnp.zeros(())

        metrics = {
            "S_norm": S_norm,
            "S_change_norm": S_change_norm,
            "memory_gate_mean": memory_gate_mean,
            "anchor_attn_mag": anchor_attn_mag,
            **write_metrics
        }

        if return_metrics:
            return loss, carry_next, metrics
        return loss, carry_next

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        memory_state: Any = None,
        anchor_state: Any = None,
        first_frame_state: Any = None,
    ) -> tuple[_model.Actions, Any] | tuple[_model.Actions, Any, at.Array]:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.config.action_horizon, self.config.action_dim))

        depth = len(self.read_proj_q)
        if memory_state is None:
            state_horizon = 1 if self.config.memory_value_type in ("vision", "vision_action") else self.config.action_horizon
            if self.config.memory_type == "lstm_cell":
                S_init = jnp.broadcast_to(self.S_init_param.value[:, None], (depth, batch_size, state_horizon, self.config.memory_rank, self.config.memory_rank))
                C_init = jnp.zeros((depth, batch_size, self.config.memory_rank, self.config.memory_rank), dtype=jnp.float32)
                memory_state = (S_init, C_init)
            else:
                memory_state = jnp.broadcast_to(self.S_init_param.value[:, None], (depth, batch_size, state_horizon, self.config.memory_rank, self.config.memory_rank))

        image_tokens = self.image_tokens(observation)
        # `first_frame_state` is what Policy captured at episode start and hands
        # back on every call. Left None -- calling the model directly -- the
        # prefix falls back to this observation's own first frame.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            observation, image_tokens=image_tokens, first_frame=first_frame_state)

        # --- anchor memory -------------------------------------------------
        # Capture the first frame from the exact SigLIP tokens used by the
        # prefix. Returning this state lets Policy retain it without a second
        # image encode; subsequent calls hand the same tensor back unchanged.
        anchor_cond = None
        if self.config.use_anchor_memory:
            if anchor_state is None:
                anchor_state = self.extract_anchor(
                    self.episode_anchor_image_tokens(
                        observation, current_image_tokens=image_tokens
                    )
                )
            anchor_cond, _ = self.read_anchor(
                self.get_vision_encoding(observation, image_tokens=image_tokens), anchor_state)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache, _, _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time, _ = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size), anchor_cond=anchor_cond
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_rep = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_rep, suffix_attn_mask], axis=-1)
            positions_suffix = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _, h_states, _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_suffix,
                kv_cache=kv_cache,
                memory_states=memory_state,
                read_fn=self.get_read_fn(memory_state, train=False),
            )
            v_t = self.action_out_proj(suffix_out[:, -self.config.action_horizon:])

            return x_t + dt * v_t, time + dt, h_states

        # Initialize dummy h_states for carry
        h_states_init = jnp.zeros((depth, batch_size, self.state_proj.out_features), dtype=prefix_tokens.dtype)
        x_0, _, final_h_states = jax.lax.while_loop(
            lambda carry: carry[1] >= -dt / 2,
            step,
            (noise, 1.0, h_states_init)
        )
        if self.config.use_anchor_memory:
            return x_0, final_h_states, anchor_state
        return x_0, final_h_states
