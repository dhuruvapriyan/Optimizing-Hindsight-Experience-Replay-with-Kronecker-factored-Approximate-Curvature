conda activate dhuruva
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/abhik/.mujoco/mujoco200/bin
export OPENAI_LOGDIR=/home/abhik/dhuruva/work/tensorboard/FetchReachSparse/HERfactoredDamping/
export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'
python -m baselines.run --alg=her --env=FetchReach-v1 --num_timesteps=1e7 --num_env=1
