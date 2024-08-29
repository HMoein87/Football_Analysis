import cv2
import os


def read_video(video_path):
    """
    Read a video file and return a list of frames.

    Parameters:
    video_path (str): The path to the video file.
    
    Returns:
    list: A list of frames
    """
    
    # Define a video capture object
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    while True:
        # Capture the video frame by frame 
        ret, frame = cap.read()
        
        if not ret:
            break;
        # Append each frame to the frames list
        frames.append(frame)
        
    return frames


def save_video(output_video_frames, output_video_path):
    """
    Save a list of frames as a video file.

    Parameters:
    output_video_frames (list): A list of frames.
    output_video_path (str): The path to save the video file.
    """
    
    # Check if the output video exists and rename it
    version = 0
    path = os.path.split(output_video_path)
    video_path = path[0]
    video_name = path[1]
        
    if os.path.exists(output_video_path):
        split_tup = os.path.splitext(video_name)
        
        # extract the file name and extension
        file_name = split_tup[0]
        file_extension = split_tup[1]
        
        while os.path.isfile(output_video_path):
            version += 1
            video_name = f'{file_name}_{version}{file_extension}'
            output_video_path = os.path.join(video_path, video_name)
            
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1],  output_video_frames[0].shape[0]))
    
    for frame in output_video_frames:
        # Write the frame to the output video file
        out.write(frame)
    
    # Release the VideoWriter object
    out.release()
    
    print(f'{video_name} released.')