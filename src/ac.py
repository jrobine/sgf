import torch
from torch import nn

import nets
import utils


class ActorCriticPolicy(nn.Module):
    """ Actor-Critic policy for discrete action spaces. """

    def __init__(self, x_dim, a_dim, actor, critic, *, compile_, device=None):
        super().__init__()
        self.actor = compile_(Actor(x_dim, a_dim, **actor, device=device))
        self.critic = compile_(nets.ScalarMLP(x_dim, **critic, device=device))

    @torch.no_grad()
    def forward(self, x, temperature=1):
        assert not self.actor.training
        a = self.actor(x, temperature=temperature)
        return a
    
    def create_trainer(self, config, *, total_its, rng, autocast, compile_):
        return ActorCriticTrainer(self, **config, total_its=total_its,
                                  rng=rng, autocast=autocast, compile_=compile_)


class Actor(nn.Module):
    """ Actor for discrete action spaces. """

    def __init__(self, x_dim, a_dim, dims, norm, act, init, out_init, device=None):
        super().__init__()
        modules, backbone_dim = nets.mlp(x_dim, dims, norm, act, init, out_bias=True, out_norm=True, device=device)
        head = nn.Linear(backbone_dim, a_dim, bias=True, device=device)
        modules.append(nets.init_(head, out_init))
        self.mlp = nn.Sequential(*modules)

    def get_stats(self, inp, temperature=1, full_precision=False):
        logits = self.mlp(inp)
        if full_precision:
            logits = logits.type(torch.float32)
        if temperature != 1:
            logits = logits / temperature
        log_probs = torch.log_softmax(logits, -1)  # normalize logits
        probs = torch.exp(log_probs)
        return log_probs, probs

    @torch.no_grad()
    def predict(self, stats):
        _, probs = stats
        sample = utils.sample_categorical(probs)
        return sample

    @torch.no_grad()
    def forward(self, x, temperature=1):
        stats = self.get_stats(x, temperature=temperature)
        sample = self.predict(stats)
        return sample

    def reinforce_loss(self, stats, a, adv, mask):
        assert self.training
        log_probs, _ = stats

        dtype = log_probs.dtype
        adv, mask = adv.type(dtype), mask.type(dtype)

        log_like = log_probs.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        reinforce_loss = mask.mean(-(adv * log_like))
        return reinforce_loss

    def entropy_loss(self, stats, mask):
        assert self.training
        log_probs, probs = stats
        log_probs = log_probs.clone()
        log_probs[probs == 0] = 0
        neg_entropy = (probs * log_probs).sum(-1)
        loss = mask.type(probs.dtype).mean(neg_entropy)
        return loss


class ActorCriticTrainer:
    """ Trainer for Actor-Critic policies. """

    def __init__(self, policy, actor_optimizer, critic_optimizer, reward_act, return_norm,
                 gamma, lmbda, entropy_coef, target_decay, target_returns, target_coef, target_every,
                 *, total_its, rng, autocast, compile_):

        self.policy = policy
        self.reward_act = nets.activation(reward_act)
        self.return_norm = ReturnNorm(device=utils.device(policy)) if return_norm else None

        self.actor_optimizer = nets.Optimizer(policy.actor, **actor_optimizer, total_its=total_its, autocast=autocast)
        self.critic_optimizer = nets.Optimizer(policy.critic, **critic_optimizer, total_its=total_its, autocast=autocast)

        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef

        if target_decay < 1 and target_every > 0:
            self.has_target = True
            self.target_critic = utils.target_network(policy.critic)
            self.target_every = target_every
            self.target_decay = target_decay
            self.target_returns = target_returns
            self.target_coef = target_coef
            self._target_it = 0
        else:
            self.has_target = False

        self.total_its = total_its
        self.rng = rng
        self.autocast = autocast

        self._optimize = compile_(self._optimize)

    def _optimize(self, x, a, target_v, ret, adv, seq_mask, it):
        # This method is compiled with torch.compile

        actor, critic = self.policy.actor, self.policy.critic

        with self.autocast():
            actor_stats = actor.get_stats(x, full_precision=True)
            entropy_loss = actor.entropy_loss(actor_stats, seq_mask)
            reinforce_loss = actor.reinforce_loss(actor_stats, a, adv, seq_mask)
            actor_loss = reinforce_loss + self.entropy_coef * entropy_loss

            metrics = {
                'reinforce_loss': reinforce_loss,
                'entropy_loss': entropy_loss,
                'actor_loss': actor_loss,
            }

        batch_size = x.shape[0]
        self.actor_optimizer.step(actor_loss, batch_size, it)

        with self.autocast():
            critic_stats = critic.get_stats(x, full_precision=True)
            return_loss = critic.loss(critic_stats, ret, seq_mask)

            if self.has_target:
                target_loss = critic.loss(critic_stats, target_v, seq_mask)
                critic_loss = return_loss + self.target_coef * target_loss
                metrics.update({
                    'return_loss': return_loss,
                    'target_loss': target_loss,
                    'critic_loss': critic_loss,
                })
            else:
                critic_loss = return_loss
                metrics['critic_loss'] = critic_loss

        self.critic_optimizer.step(critic_loss, batch_size, it)
        return metrics

    def train(self, it, xs, final_x, as_, next_rs, next_terms):
        policy = self.policy
        return_norm = self.return_norm
        critic = policy.critic

        critic.eval()
        with self.autocast():
            with torch.no_grad():
                vs = utils.apply_seq(critic, xs)

                if self.has_target:
                    self.target_critic.eval()
                    target_vs = utils.apply_seq(self.target_critic, xs)

                if self.has_target and self.target_returns:
                    final_target_v = self.target_critic(final_x)
                    next_vs = torch.cat([target_vs[1:], final_target_v.unsqueeze(0)], 0)
                    baselines = target_vs  # also use target baseline
                else:
                    # reuse the same values; this is actually not correct, since next_x != x when next_term is True
                    # however, this should not be a problem since `masks` is zero when next_term is True
                    final_v = critic(final_x)
                    next_vs = torch.cat([vs[1:], final_v.unsqueeze(0)], 0)
                    baselines = vs

                next_masks = utils.get_mask(1 - next_terms.to(next_rs.dtype))
                seq_masks = next_masks.cumulative_sequence(shifted=True)
                next_gammas = next_masks.values * self.gamma
                rets = utils.lambda_return(self.reward_act(next_rs), next_vs, next_gammas, self.lmbda)

                if return_norm is not None:
                    return_norm.update(rets)
                    adv = return_norm(rets) - return_norm(baselines)
                else:
                    adv = rets - baselines

                x, a, ret, adv, seq_mask = [utils.flatten_seq(val) for val in (xs, as_, rets, adv, seq_masks)]
                if self.has_target:
                    target_v = utils.flatten_seq(target_vs)

                batch_metrics = {
                    'rewards': next_rs.mean(),
                    'terminals': next_terms.float().mean(),
                    'values': vs.mean(),
                    'advantages': adv.mean(),
                    'returns': rets.mean(),
                }

        policy.train()
        metrics = self._optimize(x, a, target_v, ret, adv, seq_mask, it)

        if self.has_target:
            while (it - self._target_it) >= self.target_every:
                utils.ema_update(critic, self.target_critic, self.target_decay)
                self._target_it += self.target_every

        metrics.update(batch_metrics)
        return metrics


# adopted from https://github.com/danijar/dreamerv3/blob/main/dreamerv3/jaxutils.py
class ReturnNorm(nn.Module):

    def __init__(self, low_percentile=5.0, high_percentile=95.0, decay=0.99, maximum=1.0, device=None):
        super().__init__()
        self.register_buffer('inv_max', torch.tensor(1 / maximum, device=device))
        self.register_buffer('q', torch.tensor([low_percentile / 100, high_percentile / 100], device=device))
        self.register_buffer('decay', torch.tensor(decay, device=device))
        self.register_buffer('low', torch.zeros(1, device=device))
        self.register_buffer('high', torch.zeros(1, device=device))
        self.register_buffer('inv_scale', torch.zeros(1, device=device))

    def update(self, ret):
        ret = ret.type(torch.float32)
        ret_low, ret_high = torch.quantile(ret.flatten(), self.q)
        decay = self.decay
        self.low.data = decay * self.low + (1 - decay) * ret_low
        self.high.data = decay * self.high + (1 - decay) * ret_high
        self.inv_scale.data = torch.maximum(self.inv_max, self.high - self.low)

    def forward(self, ret):
        return (ret - self.low) / self.inv_scale
