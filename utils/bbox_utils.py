def get_bbox_center(bbox):
    '''
    Gets the center of the bounding box

    Args:
      bbox: bounding box
    Returns
      x: x coordinate of the center
      y: y coordinate of the center
    '''
    
    x1, y1, x2, y2 = bbox
    
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    '''
    Gets the width of the bounding box

    Args:
      bbox: bounding box
    Returns
      width: width of the bounding box
    '''
    
    return bbox[2]-bbox[0]

def measure_distance(p1, p2):
    '''
    Measures the distance between two points

    Args:
      p1: first point
      p2: second point
    Returns
      distance: distance between the two points
    '''
    
    return((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5