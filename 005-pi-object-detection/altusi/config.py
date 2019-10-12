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

PERSON_DET_XML = os.path.join(MODEL_DIR, 'person-detection-retail-0002-fp16.xml')
PERSON_DET_BIN = os.path.join(MODEL_DIR, 'person-detection-retail-0002-fp16.bin')

PERSON_DET_XML = os.path.join(MODEL_DIR, 'person-detection-retail-0013-fp16.xml')
PERSON_DET_BIN = os.path.join(MODEL_DIR, 'person-detection-retail-0013-fp16.bin')
