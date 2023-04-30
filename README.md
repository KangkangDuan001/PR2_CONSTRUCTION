# Multiple agents reinforcement learning based control of PR2 robot for construction tasks
This project aims to apply multiple agents reinforcement learning in construction tasks. Both collaboration and competition mechanisms are included in this project. The implementation of multi-agent PPO algorithm is based on the PPO algorithm from [Open AI Baselines](https://github.com/openai/baselines), [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), and [pytorch version ppo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail). 
## Installation
1. Clone the project
```
git clone https://github.com/KangkangDuan001/PR2_CONSTRUCTION.git
```
2. The inverse kinematics part is based on the ikfast generator of [pybullet-planning](https://github.com/caelan/pybullet-planning) project.
3. Create the virtual environment
```
python -m venv pr2
source pr2/bin/activate
pip install pybullet==3.2.5
pip install gym==0.21.0
pip install numpy==1.24.2
pip install torch==2.0.0
pip install attrdict==2.0.1
pip install h5py==3.8.0
pip install scipy==1.10.1
```
### Training
```
python PR2_CONSTRUCTION/train.py --env-name "pr2" --use-gae --log-interval 1 --num-steps 2000 --lr 5e-4 --entropy-coef 0.01 --value-loss-coef 0.5 --ppo-epoch 4 --num-mini-batch 5 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 2500000 --use-linear-lr-decay 
```
### Testing
```
python demo.py --load-dir trained_models/ppo
```

