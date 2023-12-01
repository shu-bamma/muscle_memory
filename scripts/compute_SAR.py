from SAR_functions import train, get_activations, find_synergies
from SAR_functions import compute_SAR, get_vid, show_video, save_manipulation_SAR
import argparse
import numpy as np


def main(args):
    #a policy is trained on an eight-object reorientation task, Reorient8-v0
    train(env_name='myoHandReorient8-v0', policy_name='play_period', timesteps=10, seed='0')
    # after training, we collect muscle activation data from policy rollouts
    muscle_data = get_activations(name='play_period', env_name='myoHandReorient8-v0', seed='0', episodes=10)
    # VAF by N synergies is computed. Here, N is selected where VAF > 0.8
    breakpoint()
    syn_dict = find_synergies(np.array(muscle_data), plot=True)
    print("VAF by N synergies:", syn_dict)
    
    if args.show_video:
        video_name = 'play_period_vid'
        get_vid(name='play_period', env_name='myoHandReorient8-v0', seed='0', episodes=1e6, video_name=video_name)
        show_video(f"{video_name}.mp4")
        
    # SAR is computed at this VAF threshold
    ica,pca,normalizer = compute_SAR(muscle_data, 20, save=False)
    # ica,pca,normalizer=load_locomotion_SAR()
    if args.file_location:
        save_manipulation_SAR(ica, pca, normalizer, args.file_location)
    

if __name__ == '__main__':
    #scomputes and saves the SAR representations in the same folder
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='myoHandReorient8-v0')
    
    parser.add_argument("--show_video", type=bool, default=False)
    
    parser.add_argument("--file_location", type=str, default=None)
    args=parser.parse_args()
    
    
    main(args)
    
    
    
    
    