"""
Central config for MAPCOS.
Update the two DATA_DIR paths once you've added the datasets in Kaggle
and can see their real mounted paths under /kaggle/input/.
"""

IMG_SIZE = 224
BATCH_SIZE = 32

DATA_DIR_VISION = "/kaggle/input/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images/data"
DATA_DIR_TABULAR = "/kaggle/input/polycystic-ovary-syndrome-pcos"

CNN_MODEL_PATH = "cnn_agent_best.h5"
LAST_CONV_LAYER = "efficientnetb0"

PCOM_THRESHOLD = 20            # Rotterdam PCOM: >=20 follicles per ovary
ROTTERDAM_CRITERIA_COUNT = 2   # 2-of-3 rule

# Follicle Counting Agent — Hough Circle params
HOUGH_PARAMS = {
    "dp": 1.2,
    "min_dist": 15,
    "param1": 50,
    "param2": 25,
    "min_radius": 3,
    "max_radius": 25,
}

# Follicle Counting Agent — watershed segmentation params
FOLLICLE_PARAMS = {
    "crop_fraction": 0.88,          # keeps top 88% of the image, drops the flat annotation bar
    "bilateral_d": 9,
    "bilateral_sigma_color": 75,
    "bilateral_sigma_space": 75,
    "min_peak_distance": 9,         # merges near-duplicate detections
    "min_area": 15,                 # loosened from 20 — was rejecting real small follicles
    "max_area": 500,
    "min_circularity": 0.4,         # loosened from 0.45 — same reason
    "border_margin": 3,             # drops crop-edge artifacts
}
