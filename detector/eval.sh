CUDA_VISIBLE_DEVICES='0' \
python eval.py \
    --dataset_file="Drone" \
    --resume="/home/zk/charac_pos/PET-main/best_checkpoint.pth" \
    --vis_dir="/home/data/ly/PET/outputs/Drone/vis_dir_6"