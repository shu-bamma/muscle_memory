"""
Example run:
python plot_result.py --path SAR-RL_successes_myoHandReorient100-v0_0/success_myoHandReorient100-v0_0.npy
"""
import argparse
import seaborn as sns
from SAR_tutorial_utils import *


def plot_success_rates(result_dir, algorithms, seeds, smth=300):
    # Define a color for each algorithm
    palette = sns.color_palette("Set2", len(algorithms))
    color_map = dict(zip(algorithms, palette))

    # labels = ['With abstraction', 'Without abstraction']
    for alg in algorithms:
        all_suc = []  # List to store success rates for all seeds

        for seed in seeds:
            file_path = os.path.join(
                result_dir,
                f"{alg}_successes_myoHandReorientOOD-v0_{seed}",
                f"success_myoHandReorientOOD-v0_{seed}.npy",
            )
            if os.path.isfile(file_path):
                suc = np.load(file_path)
                suc = smooth(suc, smth)[:-smth]
                all_suc.append(suc)
        
        # Compute the minimum length of success rates
        min_len = min([len(suc) for suc in all_suc])
        # Cut the success rates to the minimum length
        all_suc = [suc[:min_len] for suc in all_suc]
        
        episodes = range(min_len)


        if all_suc:
            all_suc = np.array(all_suc)
            avg_suc = np.mean(all_suc, axis=0)
            std_suc = np.std(all_suc, axis=0)

            alg_color = color_map[alg]
            sns.lineplot(x=episodes, y=avg_suc, color=alg_color, label=alg)
            plt.fill_between(
                episodes,
                avg_suc - std_suc,
                avg_suc + std_suc,
                color=alg_color,
                alpha=0.3,
            )

    plt.title("Success comparison on Reorient100", size=17)
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.legend(prop = { "size": 10 })
    plt.savefig("success_comparison.pdf")
    plt.show()


# Example usage
# algorithms = ["RL-E2E", "SAR-RL_scratch"]
algorithms = ["RL-SAR", "RL-E2E", "Retrieval_RL_no_SAR", "Retrieval_RL_no_SAR_flat"]
# algorithms = ["RL-E2E", "RL-E2E-Org", "RL-SAR"]
# algorithms = ["Retrieval_SAR-RL", "Retrieval_SAR-RL_flat"]
seeds = [42, 123, 751, 1001, 2001]  # Example seed numbers


def main(args):
    # Path: Success
    # plot_results(experiment="manipulation", path=args.path)

    plot_success_rates(args.result_dir, algorithms, seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
