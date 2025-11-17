# PPHAU YOLO + KMeans Segmentation Setup

   - `hw2_yolo.py` reads the extracted frames.  
   - Loads YOLO model (`yolov8n.pt`) to detect objects in RGB frames.  
   - Converts Depth → XYZ coordinates using camera intrinsics.  
   - Combines color and XYZ features to run **KMeans clustering** for foreground/background segmentation.  
   - Produces:
     - Binary mask images
     - Colored overlay images  
   - Outputs saved in `results`
