import cv2


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
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1],  output_video_frames[0].shape[0]))
    
    for frame in output_video_frames:
        # Write the frame to the output video file
        out.write(frame)
    
    # Release the VideoWriter object
    out.release()