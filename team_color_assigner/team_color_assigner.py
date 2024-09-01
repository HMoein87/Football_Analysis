from sklearn.cluster import KMeans


class TeamColorAssigner:
    '''
    Assigns teams to players
    '''
    def __init__(self):
        '''
        Initializes the team color assigner
        '''
        self.kmeans = None
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        '''
        Gets the clustering model for the image
        
        Args:
            image: image to cluster
        Returns
            kmeans: clustering model
        '''
      
        # Reshape image to 2D array
        image_2d = image.reshape(-1,3)
        
        # Perform kmeans with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        
        
        return kmeans        
        
    
    def get_player_color(self, frame, bbox):
        '''
        Gets the color of the player
        
        Args:
            frame: frame to cluster
            bbox: bounding box of the player
        Returns
            player_color: color of the player
        '''
      
        player_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        player_top_half_image = player_image[0:int(player_image.shape[0]/2), :]
        
        # Get clustering model
        kmeans = self.get_clustering_model(player_top_half_image)
        
        # Get the cluster labels for each pixels
        labels = kmeans.labels_
        
        # Reshape the labels to the  image shape
        clustered_image = labels.reshape(player_top_half_image.shape[0], 
                                         player_top_half_image.shape[1])
        
        # Get the player cluster
        corner_clusters = [clustered_image[0,0], 
                          clustered_image[0,-1], 
                          clustered_image[-1,0], 
                          clustered_image[-1,-1]]
        
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        
        player_color = kmeans.cluster_centers_[player_cluster]
        
        
        return player_color
        
        
    def assign_team_color(self, frame, player_detections):
        '''
        Assigns teams to players
        
        Args:
            frame: frame to cluster
            player_detections: dictionary of player detections
        '''
        
        # Get the player colors
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
            
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        
        self.kmeans = kmeans
        
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        
        
    def get_player_team(self, frame, player_bbox, player_id):
        '''
        Gets the team of the player
        
        Args:
            frame: frame to cluster
            player_bbox: bounding box of the player
            player_id: id of the player
        Returns
            team_id: id of the team
        '''
        
        # Check if the player is already in the dictionary
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        # Get the player color
        player_color  = self.get_player_color(frame, player_bbox)
        
        # Get the team id
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1
        
        self.player_team_dict[player_id] = team_id
        
        
        return team_id
        