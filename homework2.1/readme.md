# PPHAU YOLO + KMeans Segmentation Setup

   - `hw2_yolo.py` reads the extracted frames.  
   - Loads YOLO model (`yolov8n.pt`) to detect objects in RGB frames.  
   - Converts Depth → XYZ coordinates using camera intrinsics.  
   - Combines color and XYZ features to run **KMeans clustering** for foreground/background segmentation.  
   - Produces:
     - Binary mask images
     - Colored overlay images  
   - Outputs saved in `results`


# Venv
cd homework2.1
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install opencv-python numpy scipy scikit-learn ultralytics

python hw2_yolo.py

when done: deactivate