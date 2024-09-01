from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import numpy as np
import pandas as pd
import os
import sys

sys.path.append('../')
from utils import get_bbox_center, get_bbox_width


class Tracker:

  def __init__(self, model_path):
    '''
    Initializes the tracker
    
    Args:
      model_path: path to the model
    '''
    self.model = YOLO(model_path)
    self.tracker = sv.ByteTrack()

  def interpolate_ball_positions(self, ball_positions):
    ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
    df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
    
    # Interpolate missing values
    df_ball_positions = df_ball_positions.interpolate()
    df_ball_positions = df_ball_positions.bfill()
    
    ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]
    
    
    return ball_positions


  def detect_frames(self, frames):
    '''
    Detects objects in a frames
    
    Args:
      frames: list of frames
    Returns
      detections: list of detections
    '''
    
    batch_size = 20
    detections = []
    for i in range(0, len(frames), batch_size):
      batch = frames[i:i+batch_size]
      batch_detections = self.model.predict(batch, conf=0.1)
      detections += batch_detections
      
    # Return detections
    return detections


  def get_object_tracks(self, frames, read_from_stubs=False, stub_path=None):
    '''
    Gets object tracks from frames

    Args:
      frames: list of frames
      read_from_stubs: whether to read from stubs
      stub_path: path to the stub
    Returns
      tracks: dictionary of tracks
    '''
    
    # Check if there is pickle file 
    if read_from_stubs and stub_path is not None and os.path.exists(stub_path):
      with open(stub_path, 'rb') as f:
        tracks = pickle.load(f)
        
        return tracks

    # Detect objects
    detections = self.detect_frames(frames)
    
    # Initialize tracks
    tracks = {
        "players": [],
        "referees": [],
        "ball": []
    }

    # Iterate over frames
    for frame_num, detection in enumerate(detections):
      # Get class names
      cls_names = detection.names
      # Invert class names
      cls_names_inverse = {v: k for k, v in cls_names.items()}

      # Convert detections to supervision
      detection_supervision = sv.Detections.from_ultralytics(detection)
      
      # Convert goalkeeper to player
      for object_ind, class_id in enumerate(detection_supervision.class_id):
        if cls_names[class_id] == 'goalkeeper':
          detection_supervision.class_id[object_ind] = cls_names_inverse['player']
      
    # Track detections
      detection_with_tracks = self.tracker.update_with_detections(
        detection_supervision
      )

      # Format tracks in dictionary
      tracks['players'].append({})
      tracks['referees'].append({})
      tracks['ball'].append({})
      
      # Iterate over tracks
      for frame_detection in detection_with_tracks:
        # Get bbox, class id and track id
        bbox = frame_detection[0].tolist()
        cls_id = frame_detection[3]
        track_id = frame_detection[4]
        
        # Add track to dictionary
        if cls_id == cls_names_inverse['player']:
          tracks['players'][frame_num][track_id] = {'bbox':bbox}
          
        if cls_id == cls_names_inverse['referee']:
          tracks['referees'][frame_num][track_id] = {'bbox':bbox}
      
      # Iterate over detections
      for frame_detection in detection_supervision:
        # Get bbox and class id
        bbox = frame_detection[0].tolist()
        cls_id = frame_detection[3]
        
        # Add ball detection to dictionary
        if cls_id == cls_names_inverse['ball']:
          tracks['ball'][frame_num][1] = {'bbox':bbox}
    
    # Save tracks to pickle file
    if stub_path is not None:
      with open(stub_path, 'wb') as f:
        pickle.dump(tracks, f)
        
          
    return tracks
  
  def draw_ellipse(self, frame, bbox, color, track_id=None):
    '''
    Draws an ellipse and triangle on the frame for players and referees

    Args:
      frame: frame to draw on
      bbox: bounding box
      color: color of the ellipse
      track_id: track id
    Returns
      frame: frame with the ellipse
    '''

    # Ellipse coordinates
    y2 = int(bbox[3])
    x_center, _ = get_bbox_center(bbox)
    width = get_bbox_width(bbox)
    
    # Draw ellipse
    cv2.ellipse(frame,          
                center = (x_center, y2),
                axes = (int(width), int(0.3*width)),
                angle = 0,
                startAngle = -45,
                endAngle = 235,
                color = color,
                thickness = 2,
                lineType = cv2.LINE_4)
    
    # Draw rectangle
    rectangle_width = 40
    rectangle_height = 20
    
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    
    y1_rect = (y2 - rectangle_height //2) + 15
    y2_rect = (y2 + rectangle_height //2) + 15
    
    if track_id is not None:
      cv2.rectangle(frame,
                    (int(x1_rect), int(y1_rect)),
                    (int(x2_rect), int(y2_rect)),
                    color,
                    cv2.FILLED)
      
      x1_text = x1_rect + 12
      if track_id > 9:
        x1_text -= 5
      if track_id > 99:
        x1_text -= 8
        
      cv2.putText(frame,
                  f"{track_id}",
                  (int(x1_text), int(y1_rect + 15)),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6,
                   2)
    
    
    return frame
  
  def draw_traingle(self, frame, bbox, color):
    y = int(bbox[1])
    x , _ = get_bbox_center(bbox)
    
    triangle_points = np.array([
      [x, y],
      [x-10, y-20],
      [x+10, y-20]
    ])
    
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
    
    
    return frame
  
  def draw_annotations(self, video_frames, tracks):
    '''
    Draws annotations on the frames

    Args:
      video_frames: list of frames
      tracks: dictionary of tracks
    Returns
      output_video_frames: list of frames with annotations
    '''

    # Initialize output video frames
    output_video_frames = []
    
    # Iterate over frames
    for frame_num, frame in enumerate(video_frames):
      frame = frame.copy()
      
      # Get players, referees and ball dictionaries
      player_dict = tracks["players"][frame_num]
      referee_dict = tracks["referees"][frame_num]
      ball_dict = tracks["ball"][frame_num]
      
      # Draw players
      for track_id, player in player_dict.items():
        color = player.get("team_color", (0,0,255))
        frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
        
      # Draw referees
      for _ , referee in referee_dict.items():
        frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))
        
      # Draw ball
      for track_id, ball in ball_dict.items():
        frame = self.draw_traingle(frame, ball["bbox"], (0,255,0))
        
        
      output_video_frames.append(frame)
      
    return output_video_frames