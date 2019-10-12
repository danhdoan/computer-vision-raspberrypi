import os


# =============================================================================
# PROJECT'S ORGANIZATION
# =============================================================================
PROJECT_BASE = '.'

#===============================================================================
# PROJECT'S PARAMETERS
#===============================================================================
FONT = os.path.join(PROJECT_BASE, 'altusi', 'Aller_Bd.ttf')

TIME_FM = '-%Y%m%d-%H%M%S'

#===============================================================================
# PROJECT'S MODELS
#===============================================================================
MODEL_DIR = 'openvino-models'
FACE_DET_XML = os.path.join(MODEL_DIR, 'face-detection-adas-0001-fp16.xml')
FACE_DET_BIN = os.path.join(MODEL_DIR, 'face-detection-adas-0001-fp16.bin')

FACE_LM_XML = os.path.join(MODEL_DIR, 'landmarks-regression-retail-0009-fp16.xml')
FACE_LM_BIN = os.path.join(MODEL_DIR, 'landmarks-regression-retail-0009-fp16.bin')
