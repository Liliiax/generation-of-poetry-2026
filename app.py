import math
from typing import Callable, Tuple

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.autograd import Function

device = "cpu"
st.markdown("""<style>
h1 { text-align: center; }
.center-text { text-align: center; color: #666; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>""", unsafe_allow_html=True)

class PScan(Function):
    @staticmethod
    def forward(ctx, A_inp, X_inp):
        A, X = A_inp.clone(), X_inp.clone()
        A, X = rearrange(A, "l b d s -> b d l s"), rearrange(X, "l b d s -> b d l s")
        A_orig = A.clone()
        PScan._forward(A, X)
        ctx.save_for_backward(A_orig, X)
        return rearrange(X, "b d l s -> b l d s")

    @staticmethod
    def backward(ctx, grad_inp: Tensor) -> Tuple[Tensor, Tensor]:
        A, X = ctx.saved_tensors
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        grad_out = rearrange(grad_inp, "b l d s -> b d l s")
        grad_out = grad_out.flip(2)
        PScan._forward(A, grad_out)
        grad_out = grad_out.flip(2)
        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_out[:, :, 1:])
        return rearrange(Q, "b d l s -> l b d s"), rearrange(grad_out, "b d l s -> l b d s")

    @staticmethod
    def _forward(A: Tensor, X: Tensor) -> None:
        b, d, l, s = A.shape
        num_steps = int(math.log2(l))
        Av, Xv = A, X
        for _ in range(num_steps):
            T = Xv.size(2)
            Av, Xv = Av[:, :, :T].reshape(b, d, T // 2, 2, -1), Xv[:, :, :T].reshape(b, d, T // 2, 2, -1)
            Xv[:, :, :, 1].add_(Av[:, :, :, 1].mul(Xv[:, :, :, 0]))
            Av[:, :, :, 1].mul_(Av[:, :, :, 0])
            Av, Xv = Av[:, :, :, 1], Xv[:, :, :, 1]
        for k in range(num_steps - 1, -1, -1):
            Av, Xv = A[:, :, 2**k - 1: l: 2**k], X[:, :, 2**k - 1: l: 2**k]
            T = 2 * (Xv.size(2) // 2)
            if T < Xv.size(2):
                Xv[:, :, -1].add_(Av[:, :, -1].mul(Xv[:, :, -2]))
                Av[:, :, -1].mul_(Av[:, :, -2])
            Av, Xv = Av[:, :, :T].reshape(b, d, T // 2, 2, -1), Xv[:, :, :T].reshape(b, d, T // 2, 2, -1)
            Xv[:, :, 1:, 0].add_(Av[:, :, 1:, 0].mul(Xv[:, :, :-1, 1]))
            Av[:, :, 1:, 0].mul_(Av[:, :, :-1, 1])


pscan: Callable[[Tensor, Tensor], Tensor] = PScan.apply

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=8, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(d_model // 16, 1)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    @staticmethod
    def _next_pow2(n):
        return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))

    def forward(self, x):
        b, l, d = x.shape
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        x_in_t = x_in.transpose(1, 2)
        x_in_t = self.conv1d(x_in_t)[:, :, :l]
        x_in = x_in_t.transpose(1, 2)
        x_in = F.silu(x_in)
        y = self.ssm(x_in)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        b, l, d_in = x.shape
        n = self.A_log.shape[1]
        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
        l_pad = self._next_pow2(l)
        if l_pad != l:
            pad = l_pad - l
            deltaA = F.pad(deltaA, (0, 0, 0, 0, 0, pad))
            deltaB_x = F.pad(deltaB_x, (0, 0, 0, 0, 0, pad))
        h = pscan(deltaA.permute(1, 0, 2, 3), deltaB_x.permute(1, 0, 2, 3))
        h = h[:, :l]
        y = (h * C.unsqueeze(2)).sum(-1)
        y = y + self.D * x
        return y


class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, d_state=8, d_conv=4, expand=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        for block, norm in zip(self.blocks, self.norms):
            h = h + block(norm(h))
        return self.fc_out(h)


class CharLSTMLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_size=256, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None):
        emb = self.embedding(x)
        out, state = self.lstm(emb, state)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, state


class CharTransformerLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_heads=4, num_layers=2, ff_dim=256,
                 dropout=0.1, max_len=512, pad_idx=0):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        emb = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        out = self.transformer_encoder(emb, mask=causal_mask)
        return self.fc_out(out)


MODEL_REGISTRY = {
    "LSTM": dict(ckpt="best_lstm.pt", cls=CharLSTMLM, has_state=True),
    "Transformer": dict(ckpt="best_transformer.pt", cls=CharTransformerLM, has_state=False),
    "Mamba": dict(ckpt="best_mamba.pt", cls=MambaLM, has_state=False),
}

def load_models():
    loaded = {}
    errors = {}
    for name, spec in MODEL_REGISTRY.items():
        try:
            ckpt = torch.load(spec["ckpt"], map_location=device, weights_only=False)
            model = spec["cls"](
                ckpt["vocab_size"], pad_idx=ckpt["char2idx"]["<pad>"], **ckpt["config"]
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            loaded[name] = dict(
                model=model,
                char2idx=ckpt["char2idx"],
                idx2char=ckpt["idx2char"],
                vocab_size=ckpt["vocab_size"],
                has_state=spec["has_state"],
            )
        except Exception as e:
            errors[name] = str(e)
    return loaded, errors



def _normalize_idx2char(idx2char):
    return {int(k): v for k, v in idx2char.items()}


@torch.no_grad()
def generate_poem(model_info, max_len=300, temperature=0.8, top_k=50, seed_text=None):
    model = model_info["model"]
    char2idx = model_info["char2idx"]
    idx2char = _normalize_idx2char(model_info["idx2char"])
    has_state = model_info["has_state"]
    vocab_size = model_info["vocab_size"]

    pad_token = getattr(model.embedding, "padding_idx", None)
    if pad_token is None:
        pad_token = char2idx.get("<pad>")

    bos_token = char2idx.get("<bos>")
    eos_token = char2idx.get("<eos>")

    forbidden_tokens = {t for t in (pad_token, bos_token) if t is not None}

    if seed_text:
        idxs = [char2idx[c] for c in seed_text if c in char2idx and char2idx[c] not in forbidden_tokens]
    else:
        idxs = []

    if not idxs:
        for cand in ("\n", " "):
            if cand in char2idx and char2idx[cand] not in forbidden_tokens:
                idxs = [char2idx[cand]]
                break

    if not idxs:
        special = {pad_token, bos_token, eos_token}
        normal_ids = [i for i in range(vocab_size) if i not in special]
        idxs = [normal_ids[0]] if normal_ids else [0]

    input_ids = torch.tensor([idxs], dtype=torch.long, device=device)
    generated = list(idxs)
    state = None

    for _ in range(max_len):
        if has_state:
            logits, state = model(input_ids[:, -1:], state)
            next_logits = logits[0, -1]
        else:
            seq = input_ids[:, -512:]
            logits = model(seq)
            next_logits = logits[0, -1]

        next_logits = next_logits / max(temperature, 1e-5)

        for t in forbidden_tokens:
            if t < next_logits.size(0):
                next_logits[t] = -float("inf")

        if top_k is not None and top_k > 0:
            top_vals, top_idx = torch.topk(next_logits, min(top_k, next_logits.size(0)))
            probs = torch.zeros_like(next_logits).scatter_(0, top_idx, F.softmax(top_vals, dim=-1))
        else:
            probs = F.softmax(next_logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1).item()

        if eos_token is not None and next_id == eos_token:
            break

        generated.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1
        )

    text = "".join(idx2char.get(i, "") for i in generated)
    return text.strip()



st.set_page_config(page_title="Генератор стихов", layout="centered")

st.markdown(
    """
    <h1 style='color:#1E3A8A; margin-bottom:0;'> Генератор стихов нейросетью</h1>
    <p style='color:#1F1F1F; font-size:17px;'>
    Данное приложение использует три обученные языковые модели (
    <b>LSTM</b>, <b>Transformer</b> и <b>Mamba</b>) для генерации стихов посимвольно.
    Выберите модель, задайте количество стихотворений и нажмите кнопку «Сгенерировать».
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

loaded_models, load_errors = load_models()

if load_errors:
    for name, err in load_errors.items():
        st.warning(f"Не удалось загрузить модель «{name}»: {err}")

if not loaded_models:
    st.error("Ни одна модель не загружена. Проверьте, что файлы best_lstm.pt, "
              "best_transformer.pt и best_mamba.pt лежат рядом с приложением.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox("Выберите модель", list(loaded_models.keys()))

with col2:
    n_poems = st.number_input("Количество стихов", min_value=1, max_value=10, value=1, step=1)

with st.expander("Дополнительные настройки генерации"):
    max_len = st.slider("Максимальная длина стихотворения (в символах)", 50, 800, 300, step=50)
generate_clicked = st.button("Сгенерировать", type="primary", use_container_width=True)

if generate_clicked:
    model_info = loaded_models[model_name]
    with st.spinner(f"Модель «{model_name}» сочиняет стихи..."):
        poems = [
            generate_poem(model_info, max_len=max_len, temperature=0.5, top_k=50)
            for _ in range(int(n_poems))
        ]

    st.divider()
    st.markdown(f"### Результаты ({model_name})")
    for i, poem in enumerate(poems, start=1):
        st.markdown(
            f"""
            <div style='background-color:#EAF2FC; border-left:4px solid #1E3A8A;
                        padding:16px 20px; border-radius:8px; margin-bottom:16px;
                        white-space:pre-wrap; color:#1F1F1F; font-size:16px; line-height:1.6;'>
            <b style='color:#1E3A8A;'>Стихотворение {i}</b><br><br>{poem}
            </div>
            """,
            unsafe_allow_html=True,
        )