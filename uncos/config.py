# IO Config
groundingdino_ckpt_dir_path = "~/.cache/uncos"
uncos_ckpt_dir_path = "~/.cache/uncos"

# Uncos Params
min_area_percentage = .0001
iou_thres = 0.7
iom_threshold = 0.7
max_depth = 3  # 1.2
intersection_thres = 500
table_inlier_thr = .01
verbose_debug = False

# Backbone Params
use_sam2 = False
sam2_ckpt_path = "../sam2/checkpoints/sam2.1_hiera_large.pt"
sam_conf_score_thr = .8 if use_sam2 else 0.88 # As used in SAM official implementation
