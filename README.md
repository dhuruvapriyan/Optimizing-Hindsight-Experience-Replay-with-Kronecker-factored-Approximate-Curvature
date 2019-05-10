# Optimizing Hindsight Experience Replay with Kronecker-factored Approximate Curvature 

 

## Abstract: 

Hindsight Experience Replay (HER) is one of the efficient algorithm to solve Reinforcement Learning tasks related to sparse rewarded environments. But bad sample efficiency and poor convergence are the drawbacks of HER.  Natural gradients solves these challenges by converging the model parameters better. It avoids taking bad actions that collapse the training performance. But this methodology requires expensive computation and thus increase in  training time. In this paper we propose a methodology “Optimizing HER with Kronecker-factored Approximation Curvature (KFAC) “. Our proposed method solves the sample efficient problem and increases success rate with better convergence. 

## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/dhuruvapriyan/Optimizing-Hindsight-Experience-Replay-with-Kronecker-factored-Approximate-Curvature.git
    cd baselines
    ```
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, 
    ```bash 
    pip install tensorflow-gpu # if you have a CUDA-compatible gpu and proper drivers
    ```
    or 
    ```bash
    pip install tensorflow
    ```
    should be sufficient. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details. 

- Install baselines package
    ```bash
    pip install -e .
    ```

### MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

## Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest
pytest
```

## Training models
Most of the algorithms in baselines repo are used as follows:
```bash
python -m baselines.run --alg=her --env=<environment_id> [additional arguments]
```
### Example 1. HER+KFAC with MuJoCo FetchReach
For instance, to train a fully-connected network controlling MuJoCo humanoid using PPO2 for 20M timesteps
```bash
python -m baselines.run --alg=her --env=FetchReach-v1 --num_timesteps=2e7
```


