import argparse
import pickle
import numpy as np
import torch
import random
from SAR_functions import load_manipulation_SAR, muscle_mem_RL
from myosuite.utils.quat_math import quat2euler, euler2mat, euler2quat
from networks import InverseDynamicsModel, ForwardPredictionModel
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



def inverse_dynamcis_training(action, ica, pca, normalizer):
    pass



def forward_prediction_training(obs, action, ica, pca, normalizer):
    pass




def main(args):
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    ica, pca, normalizer = load_manipulation_SAR(args.sar_path)

    with open(args.data_path, "rb") as f:
        buffer = pickle.load(f)

    # breakpoint()
    num_samples = int(buffer.observations.shape[0] * args.percent / 100)
    # Without replacement.
    indices = np.random.choice(buffer.observations.shape[0], num_samples, replace=False)
    samples = buffer._get_samples(indices)

    observations = samples.observations.cpu().numpy()
    actions = samples.actions.cpu().numpy()[:, :39]  # Without original SAR.
    # Compute SAR.
    exit()
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

    policy_name = args.policy_name
    syn_nosyn = False if args.no_sar else True
    retrieval_env = False if args.no_retrieval else True

    muscle_mem_RL(
        env_name=args.env,
        policy_name=policy_name,
        timesteps=3e6,
        seed=str(args.seed),
        ica=ica,
        pca=pca,
        normalizer=normalizer,
        phi=0.66,
        lookup_key=observations,
        lookup_sar=sar,
        syn_nosyn=syn_nosyn,
        retrieval_env=retrieval_env
    )

    










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sar_path",
        type=str,
        default="/home/shubham/myo-dex/muscle_memory/myosuite/myosuite/agents/SAR_pretrained/manipulation",
        help="absolute path to the trained SAR model directoty",
    )
    parser.add_argument(
        "--percent",
        type=int,
        help="How many percent of the data to use for SAR.",
        required=True,
    )
    # parser.add_argument(
    #     "--policy_name",
    #     required=True
    # )
    parser.add_argument(
        "--no_sar",
        action="store_true"
    )
    # parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="/home/shubham/myo-dex/SAR-RL_buffer_myoHandReorient100-v0_0.pkl")
    parser.add_argument("--use_sar", action="store_true", default=True)
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device number (e.g., '0' or '0,1' for multiple devices)",
    )
    args = parser.parse_args()
    main(args)
