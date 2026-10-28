"""CUDA-graph the Qwen text step of RvChat (the 32 Qwen layers on one token).

Eager, a text token is ~6,200 small kernel launches (CPU-bound: ~50 ms vs ~26 ms of GPU time). Here the step is
captured once as a CUDA graph. The growing KV cache of the full-attention layers is replaced, for the step, by a
static buffer written at the current position (index_copy_ from a device-side position) and read through a mask
built in-graph; the linear-attention layers already update their states in place, so they graph as they are.
Each conversation's prefill cache is copied into the graph's persistent state when the conversation starts.
"""
import copy, types
import torch

MAXLEN = 4096  # positions the static KV buffers hold; beyond it we fall back to the eager step


class _KVProxy:
    """The cache the decoder layers see: static K/V for full-attention layers, the real cache for everything else."""
    def __init__(self, base, K, V, pos):
        self._base, self.K, self.V, self.pos = base, K, V, pos

    def update(self, k, v, layer_idx, *a, **kw):
        K, V = self.K[layer_idx], self.V[layer_idx]
        K.index_copy_(2, self.pos, k)
        V.index_copy_(2, self.pos, v)
        return K, V

    def __getattr__(self, name):
        return getattr(self._base, name)


def install(chat):
    eager = chat._qwen_step
    cfg = chat.model.config
    layers = chat.lm.layers
    full = [i for i in range(chat.n_chip, len(layers)) if cfg.layer_types[i] == "full_attention"] \
        if hasattr(cfg, "layer_types") and len(cfg.layer_types) == len(layers) else None
    if full is None:  # layer_types covers Qwen's 32 layers only: index from 0 within Qwen's stack
        lt = cfg.layer_types
        full = [chat.n_chip + j for j, t in enumerate(lt) if t == "full_attention"]
    state = types.SimpleNamespace(graph=None, src=None, pc=None)

    def _alloc(cache):
        dev, dt = chat.dev, torch.bfloat16
        state.K, state.V = {}, {}
        for i in full:
            lay = cache.layers[i]
            k = lay.keys
            state.K[i] = torch.zeros(k.shape[0], k.shape[1], MAXLEN, k.shape[3], dtype=k.dtype, device=dev)
            state.V[i] = torch.zeros_like(state.K[i])
        state.g_hc = torch.zeros(cfg.hidden_size, dtype=dt, device=dev)
        state.g_pos = torch.zeros(1, dtype=torch.long, device=dev)
        state.g_ar = torch.arange(MAXLEN, device=dev)
        state.g_logits = None

    def _body():
        h = state.g_hc.view(1, 1, -1)
        p4 = state.g_pos.view(1, 1, 1).expand(4, 1, 1).contiguous()
        pe = chat.lm.rotary_emb(h, p4[1:])
        mask = (state.g_ar <= state.g_pos).view(1, 1, 1, MAXLEN)
        proxy = _KVProxy(state.pc, state.K, state.V, state.g_pos)
        for i in range(chat.n_chip, len(layers)):
            h = layers[i](h, position_embeddings=pe, attention_mask=mask if i in full else None,
                          position_ids=p4[0], past_key_values=proxy)
        out = chat.model.lm_head(chat.lm.norm(h))[0, -1]
        if state.g_logits is None:
            state.g_logits = torch.zeros_like(out)
        state.g_logits.copy_(out)

    @torch.no_grad()
    def _init(cache, pos):
        """Point the graph's persistent state at this conversation: linear states + full-attention KV."""
        if state.graph is None:
            state.pc = copy.deepcopy(cache)  # the graph's own cache; its state tensors keep their addresses
            _alloc(cache)
            _sync(cache, pos)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    _body()
            torch.cuda.current_stream().wait_stream(s)
            state.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(state.graph):
                _body()
        _sync(cache, pos)
        state.src = cache

    @torch.no_grad()
    def _sync(cache, pos):
        for i in range(len(layers)):
            src, dst = cache.layers[i], state.pc.layers[i]
            if i in full:
                state.K[i].zero_(); state.V[i].zero_()
                state.K[i][:, :, :src.keys.shape[2]].copy_(src.keys)
                state.V[i][:, :, :src.values.shape[2]].copy_(src.values)
            if i >= chat.n_chip and hasattr(src, "conv_states") and i not in full:
                for a in ("conv_states", "recurrent_states"):
                    s_, d_ = getattr(src, a), getattr(dst, a)
                    for j in range(len(s_)):
                        if s_[j] is not None:
                            d_[j].copy_(s_[j])
                dst.has_previous_state[0] = True

    @torch.no_grad()
    def step(self, hc, cache, pos):
        if pos >= MAXLEN:
            return eager(hc, cache, pos)
        if state.src is not cache:
            _init(cache, pos)
        state.g_hc.copy_(hc)
        state.g_pos.fill_(pos)
        state.graph.replay()
        return state.g_logits.clone()

    chat._qwen_step = types.MethodType(step, chat)
    chat._qwen_step_eager = eager
    return state


def compile_chip(chat):
    """torch.compile the decode-time chip layers (the CUDA graph is captured around them). Prefill keeps the eager
    layers: it runs at a different sequence length per prompt and would recompile each time."""
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 256
    clayers = [torch.compile(chat.cm.layers[L], dynamic=False) for L in range(chat.n_chip)]

    @torch.no_grad()
    def _chip_body(self):
        h = self.cm.embed_tokens(self.g_ids)
        for L in range(self.n_chip):
            h = clayers[L](h, position_embeddings=None, attention_mask=None, position_ids=None,
                           past_key_values=self.ccache)
        self.g_out.copy_(h)
    chat._chip_body = types.MethodType(_chip_body, chat)
