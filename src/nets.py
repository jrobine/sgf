import functools
import math

import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F
import torchvision
from torch import nn, optim

import utils


torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2


class LayerNorm(nn.LayerNorm):
    # Computes channel-wise layer norm for 2d inputs.
    # Converts to float32 even when input is float16 for numerical stability.
    # (see https://github.com/pytorch/pytorch/issues/66707)
    # Uses higher epsilon value by default.

    def __init__(self, dim, eps=1e-3, device=None):
        super().__init__((dim,), eps, device=device)

    def forward(self, x):
        if x.ndim == 4:
            u = x.mean(1, keepdim=True)
            xmu = x - u
            s = xmu.square().mean(1, keepdim=True)
            x = xmu / torch.sqrt(s + self.eps)
            out = self.weight[:, None, None] * x + self.bias[:, None, None]
        else:
            out = super().forward(x)
        return out


_linear_modules = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
_norm_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, LayerNorm, nn.GroupNorm)


def _norm_init_(mod):
    if mod.weight is not None:
        nn.init.constant_(mod.weight, 1.0)
    if mod.bias is not None:
        nn.init.constant_(mod.bias, 0.0)


def zero_init_(mod):
    if isinstance(mod, _linear_modules):
        if mod.weight is not None:
            nn.init.zeros_(mod.weight)
        if mod.bias is not None:
            nn.init.zeros_(mod.bias)
    elif isinstance(mod, _norm_modules):
        _norm_init_(mod)


def normal_init_(mod, std=1.0):
    if isinstance(mod, _linear_modules):
        if mod.weight is not None:
            nn.init.normal_(mod.weight, std)
        if mod.bias is not None:
            nn.init.zeros_(mod.bias)
    elif isinstance(mod, _norm_modules):
        _norm_init_(mod)


def truncated_normal_init_(mod, scale=1.0):
    # adopted from https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/conv.py
    # used in Impala (https://github.com/google-deepmind/scalable_agent)

    if isinstance(mod, _linear_modules):
        if mod.weight is not None:
            fan_in = nn.init._calculate_correct_fan(mod.weight, mode='fan_in')
            std = math.sqrt(scale / fan_in)
            nn.init.trunc_normal_(mod.weight)
            with torch.no_grad():
                mod.weight *= std
        if mod.bias is not None:
            nn.init.zeros_(mod.bias)
    elif isinstance(mod, _norm_modules):
        _norm_init_(mod)


# adopted from https://github.com/mila-iqia/spr/blob/release/src/models.py
class RandomIntensity(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, 1, 1), device=x.device)
        noise = noise.clamp(-2.0, 2.0)
        return x * (1.0 + (self.scale * noise))


class RandomShift(nn.Module):

    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    @torch.no_grad()
    def forward(self, x):
        shift = self.shift
        n, _, h, w = x.shape
        x = nn.functional.pad(x, (shift, shift, shift, shift), mode='replicate')
        w1 = np.random.randint(0, shift + 1, n)
        h1 = np.random.randint(0, shift + 1, n)
        out = []
        for i in range(n):
            w11 = w1[i]
            h11 = h1[i]
            out.append(x[i, :, h11:h11 + h, w11:w11 + w])
        out = torch.stack(out, 0)
        return out


class Sign(nn.Module):

    def forward(self, x):
        return torch.sign(x)


class Clip(nn.Module):

    def forward(self, x):
        return torch.clip(x, -1, 1)


_acts = {
    'none': lambda: None,
    'relu': lambda: nn.ReLU(),
    'silu': lambda: nn.SiLU(),
    'sign': lambda: Sign(),
    'clip': lambda: Clip(),
}

_norms_1d = {
    'none': lambda d, device=None: None,
    'batch': lambda d, **kwargs: nn.BatchNorm1d(d, **kwargs),
    'layer': lambda d, **kwargs: LayerNorm(d, **kwargs),
}

_norms_2d = {
    'none': lambda d, device=None: None,
    'batch': lambda d, **kwargs: nn.BatchNorm2d(d, **kwargs),
    'layer': lambda d, **kwargs: LayerNorm(d, **kwargs),
}

_inits = {
    'none': lambda mod: None,
    'zeros': zero_init_,
    'normal': normal_init_,
    'truncated_normal': truncated_normal_init_,
}

_transforms = {
    'none': lambda: None,
    'random_shift': lambda **kwargs: RandomShift(**kwargs),
    'random_intensity': lambda **kwargs: RandomIntensity(**kwargs),
}


def _get_factory(factories, value, key_id='type'):
    if isinstance(value, str):
        key = value
        kwargs = {}
    elif isinstance(value, dict):
        kwargs = dict(value)
        key = kwargs.pop(key_id)
    else:
        raise ValueError(type(value))
    return functools.partial(factories[key], **kwargs)


def _build_module(factories, value, *args, none_identity=True, key_id='type', extra_kwargs=None):
    factory = _get_factory(factories, value, key_id)
    if extra_kwargs is None:
        extra_kwargs = dict()
    result = factory(*args, **extra_kwargs)
    if result is None and none_identity:
        result = nn.Identity()
    return result


def _build_modules(factories, values, args=None, none_identity=True, key_id='type', extra_kwargs=None):
    if values is None:
        return nn.Identity() if none_identity else None
    modules = []
    for i in range(len(values)):
        mod = _build_module(factories, values[i], *(args[i] if args is not None else tuple()), none_identity=False,
                            key_id=key_id, extra_kwargs=extra_kwargs[i] if extra_kwargs is not None else None)
        modules.append(mod)
    modules = [mod for mod in modules if mod is not None]
    if len(modules) == 0:
        return nn.Identity() if none_identity else None
    return modules[0] if len(modules) == 1 else nn.Sequential(*modules)


def activation(act, none_identity=True):
    return _build_module(_acts, act, none_identity=none_identity)


def norm_1d(norm, d, device=None, none_identity=True):
    return _build_module(_norms_1d, norm, d, none_identity=none_identity, extra_kwargs={'device': device})


def norm_2d(norm, d, device=None, none_identity=True):
    return _build_module(_norms_2d, norm, d, none_identity=none_identity, extra_kwargs={'device': device})


def init_(mod, init):
    initializer = _get_factory(_inits, init)
    mod.apply(initializer)
    return mod


def augmentation(transforms, none_identity=True):
    return _build_modules(_transforms, transforms, none_identity=none_identity)


def parse_dims(dims):
    if isinstance(dims, str):
        dims = tuple(int(d) for d in dims.split('-'))
    elif isinstance(dims, int):
        dims = (dims,)
    return dims


def mlp(in_dim, dims, norm, act, init, out_bias, out_norm, device=None):
    dims = parse_dims(dims)
    modules = []

    for i, dim in enumerate(dims):
        norm_mod = norm_1d(norm, dim, device=device, none_identity=False)
        act_mod = activation(act, none_identity=False)
        bias = True

        if i == len(dims) - 1:
            bias = out_bias
            if not isinstance(out_norm, bool):
                norm_mod = norm_1d(out_norm, dim, device=device, none_identity=False)
            elif not out_norm:
                norm_mod = None

        linear_mod = nn.Linear(in_dim, dim, bias=bias and (norm_mod is None), device=device)

        modules.extend([mod for mod in [linear_mod, norm_mod, act_mod] if mod is not None])
        in_dim = dim

    modules = [init_(mod, init) for mod in modules]
    out_dim = in_dim if len(modules) == 0 else dims[-1]
    return modules, out_dim


def cnn(in_dim, in_res, out_dim, channels, kernels, strides, paddings, norm, act, init,
        out_bias, out_norm, *, memory_format=None, device=None):

    channels, kernels, strides, paddings = [parse_dims(x) for x in (channels, kernels, strides, paddings)]
    num_layers = len(channels)
    if len(kernels) != num_layers or len(strides) != num_layers or len(paddings) != num_layers:
        raise ValueError('Number of channels, strides, kernels and paddings must be equal')

    modules = []
    prev_dim = in_dim
    out_res = in_res

    for i in range(num_layers):
        dim = channels[i]
        norm_mod = norm_2d(norm, dim, device=device, none_identity=False)
        act_mod = activation(act, none_identity=False)
        conv_mod = nn.Conv2d(
            prev_dim, dim, kernels[i], strides[i], paddings[i], bias=norm_mod is None, device=device)

        modules.extend([conv_mod, norm_mod, act_mod])
        out_res = int((out_res + 2 * paddings[i] - kernels[i]) / strides[i] + 1)
        prev_dim = dim

    flat_dim = prev_dim * out_res * out_res

    if not isinstance(out_norm, bool):
        norm_mod = norm_1d(out_norm, out_dim, device=device, none_identity=False)
    elif out_norm:
        norm_mod = norm_1d(norm, out_dim, device=device, none_identity=False)
    else:
        norm_mod = None

    linear_mod = nn.Linear(flat_dim, out_dim, bias=out_bias and norm_mod is None, device=device)
    modules.extend([nn.Flatten(), linear_mod, norm_mod])

    modules = [init_(mod, init) for mod in modules if mod is not None]
    if memory_format is not None:
        modules = [mod.to(memory_format=memory_format) for mod in modules]
    return modules


def transpose_cnn(in_dim, out_dim, in_res, channels, kernels, strides, paddings, norm, act, init,
                  *, memory_format=None, device=None):

    channels, kernels, strides, paddings = [parse_dims(x) for x in (channels, kernels, strides, paddings)]
    num_layers = len(channels)
    if len(kernels) != num_layers or len(strides) != num_layers or len(paddings) != num_layers:
        raise ValueError('Number of channels, strides, kernels and paddings must match')

    linear_mod = nn.Linear(in_dim, channels[0] * in_res * in_res, bias=True, device=device)
    unflatten_mod = nn.Unflatten(1, (channels[0], in_res, in_res))
    modules = [linear_mod, unflatten_mod]

    for i in range(num_layers - 1):
        norm_mod = norm_2d(norm, channels[i + 1], device=device, none_identity=False)
        conv_mod = nn.ConvTranspose2d(channels[i], channels[i + 1], kernels[i], strides[i], paddings[i],
                                        bias=norm_mod is None, device=device)
        act_mod = activation(act, none_identity=False)
        modules.extend([conv_mod, norm_mod, act_mod])

    conv_mod = nn.ConvTranspose2d(
        channels[-1], out_dim, kernels[-1], strides[-1], paddings[-1], bias=True, device=device)
    modules.append(conv_mod)

    modules = [init_(mod, init) for mod in modules if mod is not None]
    if memory_format is not None:
        modules = [mod.to(memory_format=memory_format) for mod in modules]
    return modules


class ScalarMLP(nn.Module):

    def __init__(self, in_dim, dims, norm, act, init, out_init, dist, pred, device=None, **kwargs):
        super().__init__()
        self.dist = dist
        self.pred = pred

        kw = kwargs
        if dist == 'mse':
            self.symlog = kw['symlog']
            out_dim = 1
        elif dist == 'bernoulli':
            out_dim = 1
        elif dist == 'twohot':
            self.symlog = kw['symlog']
            num_bins, low, high = kw['num_bins'], kw['low'], kw['high']
            out_dim = num_bins
            bins = utils.bins(low, high, num_bins, device=device)
            self.register_buffer('bins', bins)
            if self.symlog:
                bins_symexp = utils.symexp(bins)
                self.register_buffer('bins_symexp', bins_symexp)
        else:
            raise ValueError(f'Unknown dist: {dist}')

        modules, backbone_dim = mlp(in_dim, dims, norm, act, init, out_bias=True, out_norm=True, device=device)
        head = nn.Linear(backbone_dim, out_dim, bias=True, device=device)
        modules.append(init_(head, out_init))

        self.mlp = nn.Sequential(*modules)

    def get_stats(self, inp, full_precision=False):
        out = self.mlp(inp)

        dist = self.dist

        if full_precision and dist in ('twohot', 'bernoulli'):
            out = out.type(torch.float32)

        if dist == 'mse':
            mean = out.squeeze(-1)
            return mean
        elif dist == 'bernoulli':
            logits = out.squeeze(-1)
            return logits
        elif dist == 'twohot':
            logits = out
            log_probs = torch.log_softmax(logits, -1)  # normalize logits
            probs = torch.exp(log_probs)
            return log_probs, probs
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def predict(self, stats):
        dist, pred = self.dist, self.pred

        if dist == 'mse':
            if pred not in ('mean', 'mode'):
                raise ValueError(f'Invalid pred "{pred}" for dist "{dist}"')
            mean = stats.detach()
            if self.symlog:
                mean = utils.symexp(mean)
            return mean

        elif dist == 'bernoulli':
            logits = stats.detach()
            probs = torch.sigmoid(logits)
            if pred == 'mean':
                return probs
            elif pred == 'mode':
                mode = probs > 0.5
                return mode
            elif pred == 'sample':
                return utils.sample_bernoulli(probs)
            else:
                raise ValueError(f'Invalid pred "{pred}" for dist "{dist}"')

        elif dist == 'twohot':
            _, probs = stats
            probs = probs.detach()
            # DreamerV3 computes the mean in symlog-space,
            # but I think it makes more sense to compute it in linear space
            bins = self.bins_symexp if self.symlog else self.bins
            if pred == 'mean':
                return (probs * bins.unsqueeze(0)).sum(-1)
            elif pred == 'mode':
                idx = torch.argmax(probs, -1)
                return bins.index_select(0, idx)
            elif pred == 'sample':
                idx = utils.sample_categorical(probs)
                return bins.index_select(0, idx)
            else:
                raise ValueError(f'Invalid pred "{pred}" for dist "{dist}"')

        else:
            raise NotImplementedError()

    def forward(self, inp):
        stats = self.get_stats(inp)
        pred = self.predict(stats)
        return pred

    def loss(self, stats, target, mask=None):
        assert self.training
        dist = self.dist

        if dist == 'bernoulli':
            logits = stats
            error = F.binary_cross_entropy_with_logits(logits, target.type(logits.dtype), reduction='none')
        else:
            if self.symlog:
                target = utils.symlog(target)

            if dist == 'mse':
                mean = stats
                target = target.type(mean.dtype)
                error = F.mse_loss(mean, target, reduction='none')
            elif dist == 'twohot':
                log_probs, _ = stats
                target = utils.two_hot(target.unsqueeze(-1), self.bins)
                log_like = (log_probs * target.type(log_probs.dtype)).sum(-1)
                error = -log_like
            else:
                raise NotImplementedError(dist)

        loss = error.mean() if mask is None else mask.mean(error)
        return loss


class VectorMLP(nn.Module):

    def __init__(self, in_dim, out_dim, dims, norm, act, init,
                 out_bias, out_norm, objective, device=None):
        super().__init__()

        if objective not in ('l1', 'l2', 'cosine', 'huber'):
            raise ValueError(f'Unknown objective: {objective}')

        self.out_dim = out_dim
        self.objective = objective

        modules, backbone_dim = mlp(in_dim, dims, norm, act, init, out_bias=True, out_norm=True, device=device)
        head_modules, _ = mlp(backbone_dim, [out_dim], norm, 'none', init, out_bias=out_bias, out_norm=out_norm, device=device)
        modules.extend(head_modules)
        self.mlp = nn.Sequential(*modules)

    def forward(self, inp):
        pred = self.mlp(inp)
        return pred

    def loss(self, pred, target, mask=None):
        if self.objective == 'l1':
            error = (pred - target).abs().mean(-1)
        elif self.objective == 'l2':
            error = (pred - target).square().mean(-1)
        elif self.objective == 'cosine':
            # cosine distance
            error = 1 - F.cosine_similarity(pred, target, -1, eps=1e-8)
        elif self.objective == 'huber':
            error = F.huber_loss(pred, target, reduction='none')
        else:
            raise NotImplementedError()

        loss = error.mean() if mask is None else mask.mean(error)
        return loss


def weight_decay_param_groups(module):
    # adopted from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    decay = set()
    no_decay = set()
    for mod_name, mod in module.named_modules():
        for param_name, param in mod.named_parameters():
            full_param_name = f'{mod_name}.{param_name}' if mod_name else param_name
            if param_name.endswith('bias') or param_name.startswith('bias_'):
                no_decay.add(full_param_name)
            elif param_name.endswith('weight') or param_name.startswith('weight_'):
                if isinstance(mod, _linear_modules):
                    decay.add(full_param_name)
                elif isinstance(mod, _norm_modules):
                    no_decay.add(full_param_name)

    # validate that we considered every parameter
    param_dict = dict(module.named_parameters())
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f'parameters {str(inter_params)} made it into both decay/no_decay sets'
    assert len(param_dict.keys() - union_params) == 0, \
        f'parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set'

    return [{'params': [param_dict[name] for name in sorted(list(decay))]},
            {'params': [param_dict[name] for name in sorted(list(no_decay))],
             'weight_decay': 0.0}]


class Optimizer:
    """ Wrapper for an optimizer that handles learning rate scheduling,
        gradient clipping, and gradient scaling (autocast). """

    def __init__(self, module, type, base_lr, end_lr, warmup_its, total_its, clip, *, autocast, **kwargs):
        self.module = module
        self.base_lr = base_lr
        self.end_lr = end_lr if end_lr is not None else base_lr
        self.warmup_its = warmup_its
        self.total_its = total_its
        self.clip = clip

        if type == 'sgd':
            optim_cls = optim.SGD
        elif type == 'adam':
            optim_cls = optim.Adam
        elif type == 'adamw':
            optim_cls = optim.AdamW
        else:
            raise ValueError(f'Unknown optimizer type: {type}')

        param_groups = weight_decay_param_groups(module)
        self.optimizer = optim_cls(param_groups, lr=torch.tensor(base_lr), **kwargs)
        ctx = autocast()
        self.scaler = torch.GradScaler(ctx.device, enabled=ctx._enabled)

    def step(self, loss, batch_size, it):
        # cosine annealing learning rate + linear warmup
        if it < self.warmup_its:
            lr = self.base_lr * ((it + 1) / self.warmup_its)
        else:
            # cosine schedule
            it -= self.warmup_its
            non_warmup_its = self.total_its - self.warmup_its
            q = 0.5 * (1 + math.cos(math.pi * it / non_warmup_its))
            lr = self.base_lr * q + self.end_lr * (1 - q)
        lr *= batch_size / 256
        # wrap in tensor to prevent recompilation
        lr = torch.tensor(lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        clip = self.clip
        if clip is not None and clip != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
