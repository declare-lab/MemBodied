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

        # Apply Query Steering
        if len(xs) > 1 and xs[1] is not None and read_fn is not None and memory_state is not None:
            # xs[1] is suffix (state + actions). First token is the state token.
            # State token hidden representation is xs[1][:, 0, :]
            h_state = xs[1][:, 0, :]
            dq, _ = read_fn(layer_idx, h_state)  # shape: (B, H, num_heads, head_dim)
            # Add correction to action query tokens (index 1 to end of suffix query)
            # In q, suffix tokens start after prefix tokens.
            prefix_len = xs[0].shape[1] if xs[0] is not None else 0
            # Action tokens are suffix tokens starting from index 1 (skip state token)
            action_start = prefix_len + 1
            q_action = q[:, action_start:, :, :]
            q_rms = jnp.sqrt(jnp.mean(jnp.square(q_action), axis=-1, keepdims=True) + 1e-12)
            q = q.at[:, action_start:, :, :].add(dq * q_rms)

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

        # Apply Output Steering
        if len(xs) > 1 and xs[1] is not None and read_fn is not None and memory_state is not None:
            h_state = xs[1][:, 0, :]
            _, do = read_fn(layer_idx, h_state)  # shape: (B, H, width)
            out_action = out[-1][:, 1:, :]
            out_rms = jnp.sqrt(jnp.mean(jnp.square(out_action), axis=-1, keepdims=True) + 1e-12)
            # Add output correction to action output tokens (index 1 to end of suffix output)
            out[-1] = out[-1].at[:, 1:, :].add(do * out_rms)

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

        # Collect state token representation at the output of this layer (input to next)
        # Suffix is index 1. First token is state token.
        h_state = xs[1][:, 0, :] if xs[1] is not None else None

        return xs, (kv_cache, h_state)


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
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], _gemma.KVCache, at.Float[at.Array, "depth b d"] | None]:
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
            # Unpack outputs: layers scan returns final xs, and scanned (kv_cache, h_states)
            embedded, (kv_cache, h_states) = self.layers(
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
        return final_out, kv_cache, h_states

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
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
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
        self.read_proj_dq = [nnx.Linear(config.memory_rank, action_expert_config.num_heads * action_expert_config.head_dim, use_bias=True, rngs=rngs) for _ in range(depth)]
        self.read_proj_do = [nnx.Linear(config.memory_rank, action_expert_config.width, use_bias=True, rngs=rngs) for _ in range(depth)]

        self.write_proj_k = [nnx.Linear(in_dim, config.memory_rank, rngs=rngs) for _ in range(depth)]
        self.write_proj_v = [nnx.Linear(value_dim, config.memory_rank, rngs=rngs) for _ in range(depth)]
        # self.write_proj_beta = [nnx.Linear(gate_in_dim, 1, rngs=rngs) for _ in range(depth)]
        self.write_proj_beta = [nnx.Linear(gate_in_dim, config.memory_rank, rngs=rngs) for _ in range(depth)]
        self.write_proj_alpha = [nnx.Linear(gate_in_dim, config.memory_rank, rngs=rngs) for _ in range(depth)]

        # Trainable initial memory state parameter
        self.S_init_param = nnx.Param(
            jax.random.normal(rngs.params() if hasattr(rngs, "params") else rngs(), (depth, 1, config.memory_rank, config.memory_rank)) * 0.02
        )

        if config.memory_value_type in ("vision", "vision_action"):
            D = paligemma_config.width
            self.pooling_proj_q = nnx.Linear(action_expert_config.width, D, rngs=rngs)
            self.pooling_proj_k = nnx.Linear(D, D, rngs=rngs)
            self.pooling_proj_v = nnx.Linear(D, D, rngs=rngs)
            
        if config.memory_type == "lstm_cell":
            self.lstm_forget = [
                nnx.Linear(action_expert_config.width, config.memory_rank, rngs=rngs)
                for _ in range(depth)
            ]

        # Store action expert attention dims for use in get_read_fn reshape
        self._action_expert_num_heads = action_expert_config.num_heads

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def get_vision_encoding(self, obs: _model.Observation) -> at.Float[at.Array, "b d"]:
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            image_tokens = jax.lax.stop_gradient(image_tokens)
            tokens.append(image_tokens)

        # Pool each camera independently, then average camera representations.
        pooled_list = []
        for image_tokens in tokens:
            state_feat = self.state_proj(obs.state)
            q = self.pooling_proj_q(state_feat)
            q = jnp.expand_dims(q, axis=1)
            k = self.pooling_proj_k(image_tokens)
            v = self.pooling_proj_v(image_tokens)
            D = q.shape[-1]
            logits = jnp.einsum("bqd,bnd->bqn", q, k) / jnp.sqrt(D)
            attn = jax.nn.softmax(logits, axis=-1)
            pooled_tokens = jnp.einsum("bqn,bnd->bqd", attn, v)
            pooled_list.append(jnp.squeeze(pooled_tokens, axis=1))
        vision_encoding = jnp.mean(jnp.stack(pooled_list, axis=0), axis=0)
            
        return vision_encoding

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        ar_mask += [True]

        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def get_read_fn(self, memory_states):
        if isinstance(memory_states, (tuple, list)):
            S_state = memory_states[0]
        else:
            S_state = memory_states

        def read_fn(layer_idx, h_state):
            # h_state shape: (B, D)
            q = jax.lax.switch(layer_idx, self.read_proj_q, h_state)
            q = jnp.tanh(q)
            q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-12)

            S = S_state[layer_idx]
            r = jnp.einsum("bhij,bj->bhi", S, q)

            dq = jax.lax.switch(layer_idx, self.read_proj_dq, r)  # shape: (B, H_state, num_heads * head_dim)
            do = jax.lax.switch(layer_idx, self.read_proj_do, r)  # shape: (B, H_state, width)

            if self.config.memory_value_type in ("vision", "vision_action"):
                dq = einops.repeat(dq, "b 1 d -> b h d", h=self.config.action_horizon)
                do = einops.repeat(do, "b 1 d -> b h d", h=self.config.action_horizon)

            # Reshape to (B, H, num_heads, head_dim) using the stored num_heads from config
            dq = einops.rearrange(dq, "b h (n hd) -> b h n hd", n=self._action_expert_num_heads)

            # Apply scaling
            scale = self.config.memory_alpha / self.config.memory_rank
            # return dq * scale, do * scale
            return (dq * scale).astype(h_state.dtype), (do * scale).astype(h_state.dtype)
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

            S_prev = S_state[l]  # shape: (B, H, r, r)

            # Keys are tanh-normalized in every reported MemBodied configuration.
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
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

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
            
            if self.config.memory_value_type in ("vision", "vision_action"):
                vision_enc_flat = self.get_vision_encoding(obs_flat)
                vision_enc = einops.rearrange(vision_enc_flat, "(b t) d -> b t d", b=batch_size, t=seq_len)
                next_vision_enc = jnp.concatenate([vision_enc[:, 1:], vision_enc[:, -1:]], axis=1)
            else:
                next_vision_enc = None

            # Compute parallel prefix tokens and mask
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs_flat)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            
            # Compute parallel prefix KV cache
            _, prefix_kv_cache_flat, _ = self.PaliGemma.llm(
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

            def step_fn(carry, t):
                # Slice observation and actions at step t
                obs_t = jax.tree.map(lambda x: x[:, t] if isinstance(x, jax.Array) or hasattr(x, "shape") else x, observation)
                act_t = actions[:, t]

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
                    carry,
                    kv_cache=kv_cache_t,
                    prefix_mask=prefix_mask_t,
                    value_input=val_t,
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
                return next_carry, (loss_t, metrics_t)

            # Scan over sequence steps
            _, (loss_seq, metrics_seq) = jax.lax.scan(step_fn, carry_init, jnp.arange(seq_len))
            # Transpose sequence loss back to match batch shape
            # jax.lax.scan returns shape (seq_len, B, H)
            loss_seq = jnp.transpose(loss_seq, (1, 0, 2))

            # Average metrics across sequence steps
            avg_metrics = jax.tree.map(jnp.mean, metrics_seq)

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
                
            if self.config.memory_value_type in ("vision", "vision_action"):
                val_input = self.get_vision_encoding(observation)
            else:
                val_input = None

            loss, _, metrics = self._forward_step(
                rng, observation, actions, carry_init, value_input=val_input, train=train, return_metrics=True
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
        train: bool = False,
        return_metrics: bool = False
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)

        if kv_cache is not None:
            assert prefix_mask is not None
            suffix_len = suffix_tokens.shape[1]
            prefix_attn_mask_rep = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
            
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            full_attn_mask = jnp.concatenate([prefix_attn_mask_rep, suffix_attn_mask], axis=-1)
            
            positions_suffix = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _, h_states = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_suffix,
                kv_cache=kv_cache,
                memory_states=carry_prev,
                read_fn=self.get_read_fn(carry_prev),
            )
        else:
            prefix_tokens, prefix_mask_val, prefix_ar_mask = self.embed_prefix(observation)
            input_mask = jnp.concatenate([prefix_mask_val, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1

            (prefix_out, suffix_out), _, h_states = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                memory_states=carry_prev,
                read_fn=self.get_read_fn(carry_prev),
            )

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # Update memory state using the value input (fallback to actions)
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

        # Calculate steering ratios (only for the last layer to minimize loop/matmul overhead during sequence scans)
        depth = len(self.read_proj_q)
        if depth > 0:
            l = depth - 1
            h_state_l = h_states[l]  # shape: (B, D)
            q_l = self.read_proj_q[l](h_state_l)
            q_l = jnp.tanh(q_l)
            q_l = q_l / jnp.maximum(jnp.linalg.norm(q_l, axis=-1, keepdims=True), 1e-12)

            S_l = S_prev[l]
            if S_l.ndim == 4:
                r_l = jnp.einsum("bij,bj->bi", S_l[:, 0], q_l)
            else:
                r_l = jnp.einsum("bij,bj->bi", S_l, q_l)

            dq_l = self.read_proj_dq[l](r_l)
            do_l = self.read_proj_do[l](r_l)

            head_dim = self.read_proj_dq[0].out_features // self._action_expert_num_heads
            width = self.read_proj_do[0].out_features

            dq_steering_ratio = jnp.mean(jnp.linalg.norm(dq_l, axis=-1)) / jnp.sqrt(head_dim)
            do_steering_ratio = jnp.mean(jnp.linalg.norm(do_l, axis=-1)) / jnp.sqrt(width)
        else:
            dq_steering_ratio = jnp.zeros(())
            do_steering_ratio = jnp.zeros(())

        metrics = {
            "S_norm": S_norm,
            "S_change_norm": S_change_norm,
            "dq_steering_ratio": dq_steering_ratio,
            "do_steering_ratio": do_steering_ratio,
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
    ) -> tuple[_model.Actions, Any]:
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

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache, _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time, _ = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_rep = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_rep, suffix_attn_mask], axis=-1)
            positions_suffix = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _, h_states = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_suffix,
                kv_cache=kv_cache,
                memory_states=memory_state,
                read_fn=self.get_read_fn(memory_state),
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
        return x_0, final_h_states
