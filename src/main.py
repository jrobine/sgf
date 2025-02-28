from argparse import ArgumentParser
from pathlib import Path

import ruamel.yaml as yaml
import torch
import wandb

import envs
import utils
from agent import Agent
from ac import ActorCriticPolicy
from wm import WorldModel
from trainer import Trainer


def main():
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, required=True, help='The device used for training')
    parser.add_argument('--game', type=str, required=True, help='The Atari game, e.g., "Breakout"')
    parser.add_argument('--seed', type=int, required=True, help='The random seed to use for reproducibility')
    parser.add_argument('--config', type=str, required=True, help='The configuration file')
    parser.add_argument('--mode', type=str, nargs='?', help='The W&B mode')
    parser.add_argument('--project', type=str, nargs='?', help='The W&B project')
    parser.add_argument('--notes', type=str, nargs='?', help='The W&B notes')
    parser.add_argument('--wm_eval', type=str, default='none', help='The type of world model evaluation, one of "none", "decoder"')
    parser.add_argument('--agent_eval', type=str, default='all', help='The type of agent evaluation, one of "none", "all", "final"')
    parser.add_argument('--amp', default=False, action='store_true', help='Whether to use automatic mixed precision')
    parser.add_argument('--compile', default=False, action='store_true', help='Whether to use torch.compile')
    parser.add_argument('--save', default=False, action='store_true', help='Whether to save the models after training')
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.YAML(typ='safe', pure=True).load(f)

    # Update the configuration with the command line arguments
    config = {**config, 'config': args.config, 'game': args.game, 'seed': args.seed,
              'wm_eval': args.wm_eval, 'agent_eval': args.agent_eval,
              'amp': args.amp, 'compile': args.compile, 'save': args.save}

    # Initialize W&B
    wandb.init(project=args.project, mode=args.mode, notes=args.notes, config=config)
    config = wandb.config

    # Setup the device, torch.autocast, and torch.compile
    device = torch.device(args.device)
    autocast = lambda: torch.autocast(device_type=device.type, enabled=config.amp)
    # The arguments of torch.compile can be set here
    compile_ = lambda mod: torch.compile(mod, dynamic=True, disable=not config.compile)

    # Initialize the environment, policy, agent, and world model
    seed = (config.seed + 17) * 13
    rng = utils.seed_everything(seed)
    env = envs.atari(config.game, make=True, **config.env)

    y_dim = config.wm['y_dim']
    a_dim = env.action_space.n
    policy = ActorCriticPolicy(y_dim, a_dim, config.policy['actor'], config.policy['critic'],
                               compile_=compile_, device=device)
    agent = Agent(policy, env.action_space, config.action_stack)

    wm = WorldModel(env.observation_space, agent.stacked_action_space, **config.wm, compile_=compile_, device=device)

    # Initialize the trainer
    trainer = Trainer(env, config.game, wm, agent, seed, **config.trainer,
                      wm_eval=config.wm_eval, agent_eval=config.agent_eval, buffer_device=device,
                      rng=rng, autocast=autocast, compile_=compile_)

    print(f'Starting... (seed: {seed})')
    print(f'World Model # params: {utils.count_params(wm)}')
    print(f'Agent # params: {utils.count_params(agent)}')

    # Train the agent and world model
    while not trainer.is_finished():
        metrics = trainer.train()

        if len(metrics) > 0:
            wandb.log(metrics, step=trainer.it)

        if trainer.it == 0:
            print('Training...')  # good, everything worked up to this point

    # Save the models, if necessary
    if config.save:
        torch.save(wm.state_dict(), Path(wandb.run.dir) / 'wm.pt')
        torch.save(agent.state_dict(), Path(wandb.run.dir) / 'agent.pt')
        wandb.save('wm.pt')
        wandb.save('agent.pt')
        if config.wm_eval == 'decoder':
            torch.save(trainer.wm_trainer.decoder.state_dict(), Path(wandb.run.dir) / 'decoder.pt')
            wandb.save('decoder.pt')

    # Cleanup
    trainer.close()
    wandb.finish()


if __name__ == '__main__':
    main()
