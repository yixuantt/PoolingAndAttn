
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

def l2norm(t):
    return F.normalize(t, dim = -1)

def default(val, d):
    return val if exists(val) else d

def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)
    
class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = torch.nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        return self.fn(x, **kwargs)
    
class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * torch.nn.functional.gelu(gates)

class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            torch.nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)

class Attention(torch.nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(context_dim, inner_dim, bias = False)
        self.to_kv = torch.nn.Linear(query_dim, inner_dim * 2, bias = False)

        self.to_out = torch.nn.Linear(inner_dim, query_dim, bias = False)

    def forward(self, x, context = None):
        h = self.heads
        q = self.to_q(context)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)
    
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim = 4096,
        latent_dim = 4096,
        cross_dim_head = 2048,
        num_cross_heads = 32,
        num_latents_value = 768,
        layers = 32,
    ):
        super().__init__()

        num_latents, latent_dim, cross_heads, cross_dim_head, dim = num_latents_value, latent_dim, num_cross_heads, cross_dim_head,hidden_dim

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.pos_emb = nn.Embedding(layers, dim)
        self.normalize = True

        # init latent_attention and latents
        self.cross_attend_blocks = torch.nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head),
                    context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim)),
        ])


    def forward(self, hiddens):
        pos_emb = self.pos_emb(torch.arange(hiddens.shape[1], device = hiddens.device))
        hiddens = hiddens + pos_emb

        cross_attn, cross_ff = self.cross_attend_blocks

        x = repeat(self.latents, 'n d -> b n d', b = hiddens.shape[0])
        x = cross_attn(hiddens, context = x) + x
        x = cross_ff(x) + x

        x = torch.mean(x, dim=1)
        if self.normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x