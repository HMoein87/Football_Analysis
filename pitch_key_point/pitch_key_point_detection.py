from inference import get_model
import supervision as sv
import numpy as np
import sys

from .soccer import SoccerPitchConfiguration
from .draw_pitch import draw_pitch


class PitchKeyPoints():
    
    def __init__(self):
        pass
    
    
    def key_point_detection(self, frames, api_key):
        
        FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"
        FIELD_DETECTION_MODEL = get_model(model_id=FIELD_DETECTION_MODEL_ID, api_key=api_key)
        CONFIG = SoccerPitchConfiguration()
        
        frames_key_points = {}
        pitch_reference_points= {}
            
        for frame_num, frame in enumerate(frames):
            result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
            key_points = sv.KeyPoints.from_inference(result)

            filter = key_points.confidence[0] > 0.5
            frame_reference_points = key_points.xy[0][filter]
            frame_reference_key_points = sv.KeyPoints(
                xy=frame_reference_points[np.newaxis, ...])
            
            frames_key_points[frame_num] = frame_reference_key_points

            reference_points = np.array(CONFIG.vertices)[filter]
            pitch_reference_points[frame_num] = reference_points
            
            
        return frames_key_points, pitch_reference_points
    
    