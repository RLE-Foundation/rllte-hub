import hydra
import os
os.environ["OMP_NUM_THREADS"] = "1"
from hsuanwu.env import (make_atari_env,
                         make_bullet_env,
                         make_dmc_env,
                         make_procgen_env,
                         make_multibinary_env
                         )
from hsuanwu.common.engine import HsuanwuEngine

@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cfgs'), config_name='config')
def main(cfgs):
    print(cfgs)
    if 'dmc_pixel' in cfgs.experiment:
        train_env = make_dmc_env(**cfgs.env)
        test_env = make_dmc_env(**cfgs.env)

    elif 'dmc_state' in cfgs.experiment and 'impala' not in cfgs.experiment:
        train_env = make_dmc_env(**cfgs.env)
        test_env = make_dmc_env(**cfgs.env)

    elif 'atari' in cfgs.experiment and 'impala' not in cfgs.experiment:
        train_env = make_atari_env(**cfgs.env)
        test_env = make_atari_env(**cfgs.env)

    elif 'bullet' in cfgs.experiment:
        train_env = make_bullet_env(**cfgs.env)
        test_env = make_bullet_env(**cfgs.env)

    elif 'procgen' in cfgs.experiment:
        train_env = make_procgen_env(**cfgs.env)
        test_env = make_procgen_env(**cfgs.env)

    elif 'multi_binary' in cfgs.experiment:
        train_env = make_multibinary_env(env_id="multibinary_pixel", device='cuda:0', num_envs=8)
        test_env = make_multibinary_env(env_id="multibinary_pixel", device='cuda:0', num_envs=8)

    elif 'impala_atari' in cfgs.experiment:
        train_env = make_atari_env(**cfgs.env)
        test_env = make_atari_env(env_id=cfgs.env.env_id,
                                  num_envs=1,
                                  seed=cfgs.env.seed,
                                  frame_stack=cfgs.env.frame_stack,
                                  distributed=cfgs.env.distributed)
        
    elif 'impala_dmc_state' in cfgs.experiment:
        train_env = make_dmc_env(**cfgs.env)
        test_env = make_dmc_env(env_id=cfgs.env.env_id,
                                num_envs=1,
                                visualize_reward=cfgs.env.visualize_reward,
                                from_pixels=cfgs.env.from_pixels,
                                device=cfgs.env.device)
    else:
        raise NotImplementedError

    engine = HsuanwuEngine(cfgs=cfgs, train_env=train_env, test_env=test_env)
    engine.invoke()

if __name__ == '__main__':
    main()