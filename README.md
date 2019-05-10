
## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
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
python -m baselines.run --alg=HER --env=<environment_id> [additional arguments]
```
### Example 1. PPO with MuJoCo Humanoid
For instance, to train a fully-connected network controlling MuJoCo humanoid using PPO2 for 20M timesteps
```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7
```


