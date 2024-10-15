srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/aishell_asr & \
srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/aishell_dev & \
srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/aishell2_ios_test & \
srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/aishell2_mic_test & \
srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/aishell2_android_test & \

# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/aac_clotho & \
# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/ClothAQA_all & \
# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/cochlscene & \

# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/librispeech_dev_clean & \
# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/librispeech_dev_other & \
# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/librispeech_test_clean & \
# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/librispeech_test_other & \

# srun -p speechllm -n 1 --gpus-per-task 1  python test_model.py --input data/internlm2_evaluate/meld & \
# srun -p speechllm -n 1 --gpus-per-task 1  python test_model.py --input data/internlm2_evaluate/NS & \
# srun -p speechllm -n 1 --gpus-per-task 1  python test_model.py --input data/internlm2_evaluate/tut2017_test & \
# srun -p speechllm -n 1 --gpus-per-task 1  python test_model.py --input data/internlm2_evaluate/VocalSound & \

# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/covost2_de-en & \
# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/covost2_en-de & \
# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/covost2_en-zh & \
# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/covost2_zh-en & \
# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/covost2_fr-en & \
# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/covost2_it-en & \
# srun -p speechllm -n 1 --gpus-per-task 1 python test_model.py --input data/internlm2_evaluate/covost2_es-en & \