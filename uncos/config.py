# IO Config
GROUNDINGDINO_CKPT_DIR_PATH = "~/.cache/uncos"
UNCOS_CKPT_DIR_PATH = "~/.cache/uncos"
SAM2_CKPT_PATH = "../segment-anything-2/checkpoints/sam2_hiera_large.pt"

# Uncos Params
MIN_AREA_PERCENTAGE = .0001
IOU_THRES = 0.7
MAX_DEPTH = 3  # 1.2
INTERSECTION_THRES = 500
TABLE_INLIER_THR = .01
VERBOSE_DEBUG = False

# Backbone Params
USE_SAM2 = False
SAM_CONF_SCORE_THR = .8 if USE_SAM2 else 0.88 # As used in SAM official implementation
