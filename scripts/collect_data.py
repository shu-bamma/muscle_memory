import argparse
from SAR_tutorial_utils import *


def main(args):
    ica, pca, normalizer = load_manipulation_SAR(args.sar_path)
    policy_name = "SAR-RL"
    env_name = "myoHandReorient100-v0"
    phi = 0.66

    frames = []
    # env = SynNoSynWrapper(gym.make(env_name), ica, pca, normalizer, phi)
    env = gym.make(env_name)
    
    camera = "front"

    is_solved = []
    total_steps = 0
    model = SAC.load(args.ckpt_path, env=env, verbose=1)

    epi = 0
    pbar = tqdm(total=args.episodes)
    # for i, __ in enumerate(tqdm(range(args.episodes))):
    while epi < args.episodes:
        env.reset()

        vec = VecNormalize.load(
            args.env_path,
            DummyVecEnv([lambda: env]),
        )

        rs = 0
        done = False
        solved = False

        observations = []
        actions = []
        rewards = []
        dones = []
        infos = []

        curr_steps = 0
        curr_frames = []
        while not done:
            curr_steps += 1
            o = vec.normalize_obs(env.get_obs())
            a, __ = model.predict(o, deterministic=False)
            # Note: It does not store scaled action. (As they do in off_policy_algorithm.py)

            if epi % args.video_intervals == 0:
                frame = env.sim.renderer.render_offscreen(
                    width=640, height=480, camera_id=camera
                )
                curr_frames.append(frame)

            next_o, r, done, info = env.step(a)
            solved = np.logical_or(solved, info["solved"])

            observations.append(next_o)
            actions.append(a)
            rewards.append(r)
            dones.append(done)
            infos.append(info)

            rs += r

        if solved:
            epi += 1
            total_steps += curr_steps
            for i in range(len(observations)):
                model._store_transition(
                    model.replay_buffer,
                    np.array([actions[i]]),
                    np.array([observations[i]]),
                    np.array([rewards[i]]),
                    np.array([dones[i]]),
                    [infos[i]],
                )
            frames.extend(curr_frames)
            pbar.update(1)

        is_solved.append(solved)

    # Save replay buffer.
    model.save_replay_buffer(f"{policy_name}_buffer_{env_name}_{args.seed}")

    print(f"Average success rate: {np.mean(is_solved)}")
    print(f"total_steps: {total_steps}")

    env.close()
    skvideo.io.vwrite(
        f"collect_data.mp4", np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"}
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
        "--device",
        type=str,
        default="0",
        help="CUDA device number (e.g., '0' or '0,1' for multiple devices)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--env_path", type=str, required=True)
    parser.add_argument(
        "--episodes", type=int, default=30000, help="Number of episodes."
    )
    parser.add_argument(
        "--video_intervals",
        type=int,
        default=1000,
        help="How often to save videos in every episode.",
    )
    args = parser.parse_args()
    main(args)
