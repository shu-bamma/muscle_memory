import argparse
import pickle
import numpy as np
import torch
import random
from SAR_functions import load_manipulation_SAR, muscle_mem_RL
from myosuite.utils.quat_math import quat2euler, euler2mat, euler2quat

import os


def set_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    ica, pca, normalizer = load_manipulation_SAR(args.sar_path)

    with open(args.data_path, "rb") as f:
        buffer = pickle.load(f)

    num_samples = int(buffer.observations.shape[0] * args.percent / 100)
    # Without replacement.
    indices = np.random.choice(buffer.observations.shape[0], num_samples, replace=False)
    samples = buffer._get_samples(indices)

    observations = samples.observations.cpu().numpy()
    actions = samples.actions.cpu().numpy()[:, :39]  # Without original SAR.
    # Compute SAR.
    if args.flat_action:
        sar = actions
    else:
        sar = normalizer.transform(ica.transform(pca.transform(actions)))
    # obs[23: 26]=obj pose; obs[26:32]=obj_vel; obs[32:38]=obj_rot+ obj_des_rot; obs[38:41]=obj_err_pos; obs[41:44]: obj_err_rot
    curr_rot = euler2mat(observations[:, 32:35])
    goal_rot = euler2mat(observations[:, 35:38])
    # Make it vector.
    curr_rot = curr_rot.reshape(-1, 9)
    goal_rot = goal_rot.reshape(-1, 9)

    observations = np.concatenate([observations[:, 23:26], curr_rot, goal_rot], axis=1)
    observations = torch.from_numpy(observations).float().cuda()
    # lookup_observation: # samples, obs_dim
    # lookup_sar: # samples, sar_dim

    policy_name = "Retrieval_SAR-RL" if not args.flat_action else "Retrieval_SAR-RL_flat"
    muscle_mem_RL(
        env_name=args.env,
        policy_name=policy_name,
        timesteps=1.5e6,
        seed=str(args.seed),
        ica=ica,
        pca=pca,
        normalizer=normalizer,
        phi=0.66,
        lookup_key=observations,
        lookup_sar=sar,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sar_path",
        type=str,
        default=None,
        help="absolute path to the trained SAR model directoty",
    )
    parser.add_argument(
        "--percent",
        type=int,
        help="How many percent of the data to use for SAR.",
        required=True,
    )
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--flat_action", action="store_true", default=False)
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device number (e.g., '0' or '0,1' for multiple devices)",
    )
    args = parser.parse_args()
    main(args)
