# muscle_memory
link to experiments and report: [muscle-memory]([https://www.openai.com](https://drive.google.com/file/d/1WoYa3lVMFu_zILt0xRwo_a26LNfe1Wy7/view)) 

## Installation
```
conda create --name muscle python=3.8
conda activate muscle 
cd myosuite 
pip install -e . 
```

### install/modify additional packages 
```
pip install joblib scikit-learn 

pip install setuptools==65.5.0 pip==21

pip install stable-baselines3==1.7 
pip install gym==0.13
```

### running experiments
- First pretrain muscle synergies [SAR](https://sites.google.com/view/sar-rl/) by runninig ```train_SAR100.py ``` followed by ``` collect_data.py``` to collect the rollouts
-  ```train_retreival.py``` runs the main retreival RL polcy (SAC with SAR action memory) along with the flat action retrieval ablation
-  ```train_SAR100.py ```  runs the other baselines including [SAR-RL](https://sites.google.com/view/sar-rl/)
- Use ```plot_result.py``` for visualisig reuslts of a single seed and ```plot_seed.py ``` for the combined results
