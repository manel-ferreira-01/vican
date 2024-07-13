"""
    dataset.py
    Gabriel Moreira
    Sep 18 2023
"""
import os
import json
import numpy as np

from .geometry import SE3
from .cam import Camera


class Dataset(object):
    def __init__(self, root: str):
        """
            Stores a dataset of images, cameras,
            object poses (optional). 

            Parameters
            ----------
            root : str 
                Path to image directory. Images follow 
                naming convention root/<timestamp>/<camera_id>.jpg
                Camera metadata should be in root/cameras.json
                (Optional) Object pose should be in
                root/object_pose_<n>.json.
        """
        self.root = root
        self.cam_path = os.path.join(root, "cameras.json")

        assert os.path.isfile(self.cam_path)

        self.read_cameras()
        self.read_im_data()
        self.read_object()


    def read_cameras(self):
        """
            Load cameras dictionary from JSON file
            generated by render.py.
        """
        # Camera dictionary indexed by camera id
        with open(self.cam_path) as f:
            data = json.load(f)

        self.cams = {}
        for k, v in data.items():
            K = np.array([[v['fx'], 0.0, v['cx']],
                          [0.0, v['fy'], v['cy']],
                          [0.0, 0.0, 1.0]])
            
            self.cams[k] = Camera(id=k,
                                  intrinsics=K,
                                  distortion=np.array([v["distortion"]], dtype=np.float64),
                                  extrinsics=SE3(R=np.array(v['R']),
                                                 t=np.array(v['t'])),
                                  resolution_x=v["resolution_x"],
                                  resolution_y=v["resolution_y"])
            

    def read_object(self):
        """
            Read object ground-truth poses from JSON files.
        """
        aruco_filenames = [fn for fn in os.listdir(self.root) \
                            if fn.split('_')[0] == "object"]

        self.object = {}
        for filename in aruco_filenames:
            with open(os.path.join(self.root, filename)) as f:
                object_data = json.load(f)
            for t, pose_dict in object_data.items():
                self.object[t] = SE3(R=np.array(pose_dict['R']),
                                     t=np.array(pose_dict['t']))


    def read_im_data(self):
        """
            Read directory containing images.
        """
        self.im_data = {"filename"  : [],
                        "timestamp" : [],
                        "cam"       : [],
                        "cam_id"    : []}
        
        timestamps = [t for t in os.listdir(self.root) if t.isnumeric() \
                      and os.path.isdir(os.path.join(self.root, t))]
        for t in timestamps:
            filenames = os.listdir(os.path.join(self.root, t))
            for filename in filenames:
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    cam_id = filename.split('.')[0]
                    self.im_data['cam_id'].append(cam_id)
                    self.im_data['filename'].append(os.path.join(self.root, t, filename))
                    self.im_data['timestamp'].append(t)
                    self.im_data['cam'].append(self.cams[cam_id])


