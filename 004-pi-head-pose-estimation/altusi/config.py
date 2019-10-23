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

pFPS = 5

#===============================================================================
# PROJECT'S MODELS
#===============================================================================
MODEL_DIR = 'openvino-models'

# face detection model
FACE_DET_XML = os.path.join(MODEL_DIR, 'face-detection-adas-0001-fp16.xml')
FACE_DET_BIN = os.path.join(MODEL_DIR, 'face-detection-adas-0001-fp16.bin')

# facial landmark model
FACE_LM_XML = os.path.join(MODEL_DIR, 'landmarks-regression-retail-0009-fp16.xml')
FACE_LM_BIN = os.path.join(MODEL_DIR, 'landmarks-regression-retail-0009-fp16.bin')

# face reidentification model
FACE_EMB_XML = os.path.join(MODEL_DIR, 'face-reidentification-retail-0095-fp16.xml')
FACE_EMB_BIN = os.path.join(MODEL_DIR, 'face-reidentification-retail-0095-fp16.bin')

# head pose estimation model
HEAD_POSE_XML = os.path.join(MODEL_DIR, 'head-pose-estimation-adas-0001-fp16.xml')
HEAD_POSE_BIN = os.path.join(MODEL_DIR, 'head-pose-estimation-adas-0001-fp16.bin')

