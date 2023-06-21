# python -m torch.distributed.launch\
#   --nproc_per_node 4\
#   train.py\
#   --data config/data/voc12.yaml\
#   --config config/models/pidnet.yaml\
#   --training config/training/pidnet.yaml

python train.py\
  --data config/data/voc12.yaml\
  --config config/models/pidnet.yaml\
  --training config/training/pidnet.yaml