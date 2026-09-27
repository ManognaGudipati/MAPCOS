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
