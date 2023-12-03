import argparse
from SAR_functions import load_manipulation_SAR, muscle_mem_RL
import os 

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    ica,pca,normalizer = load_manipulation_SAR(args.sar_path)
    # SAR is used to train on a 100-object reorientation task, Reorient100-v0
    #TODO: convert buffer data to lookup_table_dict 
    #.....
    #.....
    lookup_table_dict={}

    muscle_mem_RL(env_name='myoHandReorient100-v0', policy_name='SAR-RL', timesteps=1.5e6, 
            seed='0', ica=ica, pca=pca, normalizer=normalizer, phi=.66, lookup_table=lookup_table_dict) 


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--sar_path", type=str, default=None,help="absolute path to the trained SAR model directoty")
    parser.add_argument("--device", type=str, default="0", help="CUDA device number (e.g., '0' or '0,1' for multiple devices)")
    args = parser.parse_args()
    main(args)

