"""
    cam.py
    Gabriel Moreira
    Sep 18, 2023
"""
import cv2 as cv
import numpy as np
import multiprocessing as mp
from functools import partial
from typing import Iterable
import time

from .geometry import SE3

class Camera(object):
    def __init__(self,
                 id: str,
                 intrinsics: np.ndarray,
                 distortion: np.ndarray,
                 extrinsics: SE3,
                 resolution_x: int,
                 resolution_y: int):
        """
            Perspective camera.

            Parameters
            ----------
            id : str
                Unique camera identifier.
            intrinsics : np.ndarray
                Intrinsics 3x3 matrix.
            distortion : np.ndarray
                Distortion vector with size 12.
            extrinsics: SE3
                Rigid transformation with camera pose
                in the world frame.
            resolution_x : int
            resolution_y : int
        """
        self.id = id
        self.intrinsics   = intrinsics.squeeze()
        self.distortion   = distortion
        self.extrinsics   = extrinsics
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

    def __repr__(self) -> str:
        repr = "Camera {}x{} id={}\n".format(self.resolution_y,
                                             self.resolution_x,
                                             self.id)
        repr += "Intrinsics:\n"
        repr += str(self.intrinsics)
        repr += "\nDistortion:\n"
        repr += str(self.distortion)
        repr += "\nExtrinsics:\n"
        repr += str(self.extrinsics)
        return repr


def gen_marker_uid(im_filename: str, marker_id: str) -> str:
    """
        Generate unique identifier for a marker 
        detected in an image.

        Parameters
        ----------
        im_filename : str
            Image file name with format 
            <timestep>/<camera_id>.jpg where the 
            marker was detected.
        m : str
            Unique identifier of the detected marker.

        Returns
        -------
        marker_uid : str
            Marker unique ID with format <timestamp>_<m>
    """
    timestamp = im_filename.split('/')[-2] 
    marker_uid = timestamp + '_' + marker_id
    return marker_uid

def estimate_pose_charuco_worker(im_filename: str,
                                 cam: Camera,
                                 target_dict: dict,
                                 flags: str,
                                 brightness: int,
                                 contrast: int) -> dict:
    
    
    start = time.time()
    
    #create board in here instead of passing it as an argument, since aruco methods are not pickable.... ~1ms
    charuco_dict = dict()
    for i in range(0,target_dict["num_boards"]+1):
        
        charuco_board = cv.aruco.CharucoBoard(
            size=(target_dict[str(i)]["sizeX"], target_dict[str(i)]["sizeY"]),
            squareLength=target_dict[str(i)]["squareLength"],
            markerLength=target_dict[str(i)]["markerLength"],
            dictionary= cv.aruco.getPredefinedDictionary(target_dict[str(i)]["dictionary"]),
            ids=target_dict[str(i)]["ids"])
        
        charuco_board.setLegacyPattern(True)
        charuco_dict[str(i)] = charuco_board        
    
    output = dict()
    
    im = cv.imread(im_filename)
    im = np.int16(im)
    
    if contrast != 0:
        im = im * (contrast/127+1) - contrast
        
    im += brightness
    im = np.clip(im, 0, 255)
    im = np.uint8(im)
    
    #extract aruco dictionary
    aruco_dict = charuco_dict["0"].getDictionary()
    
    # Detect aruco markers
    corners, ids, rejected = cv.aruco.detectMarkers(im, aruco_dict)
    
    if len(corners) == 0:
        return output
    
    #go through all the charuco boards and compute one edge for each
    for board_id, charuco_board in charuco_dict.items():
        
        if board_id == "detector":
            continue
        
        flag, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(
                    corners, ids, im, charuco_board)
        
        if flag:
            
            objPoints, imPoints = charuco_board.matchImagePoints(charuco_corners, charuco_ids)
            
            if len(objPoints) < 4:
                continue # no enough points to estimate pose
            
            retval, rvec, tvec = cv.solvePnP(objPoints, imPoints,
                                              cam.intrinsics,
                                              cam.distortion,
                                              flags=eval('cv.' + flags))
            
            # TODO: check if pnp refinement is necesseary and adequeate for charuco boards
            if retval:
            
                R = cv.Rodrigues(rvec)[0]
                pose= SE3(R=R, t=tvec)
                
                
                
                reprojected = cv.projectPoints(objPoints, R, tvec,
                                                cam.intrinsics, cam.distortion )[0]
                
                #unpack impoints
                reprojection_err = np.linalg.norm(reprojected - imPoints, axis=1).max()
                key = (cam.id, gen_marker_uid(im_filename,board_id))
                
                output[key] = {'pose' : pose,
                            'corners' : imPoints.squeeze(),
                            'reprojected_err' : reprojection_err,
                            'im_filename' : im_filename,
                            'distance' : np.linalg.norm(tvec),
                            'time' : time.time()-start}
            
            
    return output

def estimate_pose_aruco_worker(im_filename: str,
                         cam: Camera,
                         target_dict: dict,
                         flags: str,
                         brightness: int,
                         contrast: int) -> dict:
    """
        Reads image from im_filename, detects arUco
        markers, estimates pose of all the detected 
        markers and returns an edge dictionary. 

        NOTE: estimate_pose_mp is the parallel version.

        Parameters
        ----------
        im_filename : str
            Image file name with format  
            <timestep>/<camera_id>.jpg.
        cam : Camera
            Camera corresponding to image im_filename.
        aruco: str
            OpenCV arUco dictionary.
        marker_size: float
            Real size of arUco markers.
        corner_refine: str
            See OpenCV corner refinement options. 
        flags: str
            Method to solve PnP - See OpenCV.
        brightness: int
            Image brightness preprocessing.
        contrast: int
            Image contrast preprocessing.

        Returns
        -------
        output : dict
            Camera-marker edge dictionary. Keys are tuples
            (<camera_id>, <timestamp>_<marker_id>).
            Values are dicts with "pose" (SE3), "corners" (np.ndarray), 
            "reprojected_err" (float) and "im_filename" (str).
    """
    # Unpack target_dict
    aruco = target_dict["dictionary"]
    marker_size = target_dict["marker_size"]
    corner_refine = target_dict["corner_refine"]
    
    dictionary = cv.aruco.getPredefinedDictionary(eval('cv.aruco.' + aruco))
    parameters = cv.aruco.DetectorParameters()
    if corner_refine is not None:
        parameters.cornerRefinementMethod = eval('cv.aruco.' + corner_refine)
    parameters.cornerRefinementMinAccuracy = 0.05
    parameters.adaptiveThreshConstant = 10
    parameters.cornerRefinementMaxIterations = 50
    parameters.adaptiveThreshWinSizeStep = 5
    parameters.adaptiveThreshWinSizeMax = 35

    im = cv.imread(im_filename)
    im = np.int16(im)

    if contrast != 0:
        im = im * (contrast/127+1) - contrast

    im += brightness
    im = np.clip(im, 0, 255)
    im = np.uint8(im)

    marker_corners, marker_ids, _ = cv.aruco.detectMarkers(im, dictionary, parameters=parameters)

    marker_points = np.array([[-1, 1, 0],
                              [1, 1, 0],
                              [1, -1, 0],
                              [-1, -1, 0]], dtype=np.float32)
    
    marker_points *= marker_size * 0.5

    output = dict()
    if len(marker_corners) > 0:
        marker_ids = list(map(str, marker_ids.flatten()))
        for corners, marker_id in zip(marker_corners, marker_ids):
            corners = corners.squeeze()

            flag, rvec, t = cv.solvePnP(marker_points,
                                        imagePoints=corners,
                                        cameraMatrix=cam.intrinsics,
                                        distCoeffs=cam.distortion,
                                        flags=eval('cv.' + flags))
            if not flag:
                continue
            rvec, t = cv.solvePnPRefineLM(marker_points,
                                          imagePoints=corners,
                                          cameraMatrix=cam.intrinsics,
                                          distCoeffs=cam.distortion,
                                          rvec=rvec,
                                          tvec=t)
            R = cv.Rodrigues(rvec)[0]
            pose = SE3(R=R, t=t)
            reprojected = cv.projectPoints(marker_points, R, t,
                                           cam.intrinsics, cam.distortion)[0].squeeze()
            
            reprojection_err = np.linalg.norm(reprojected - corners, axis=1).max()
            key = (cam.id, gen_marker_uid(im_filename, marker_id))

            output[key] = {'pose' : pose,
                           'corners' : corners.squeeze(), 
                           'reprojected_err' : reprojection_err,
                           'im_filename' : im_filename}
        return output
    


def estimate_pose_mp(im_filenames: Iterable[str],
                     cams: Iterable[Camera],
                     target_dict: dict,
                     brightness: int,
                     contrast: int,
                     flags: str,
                     ) -> dict:
    """
        Multiprocessing pool of estimate_pose_worker. 
        Iterates through all image filenames provided in im_filenames,
        detects arUco markers and returns edge dictionary. 
        Keys are (<camera_id>, <timestamp>_<marker_id>)
        Values are dicts with "pose" (SE3), "corners" (np.ndarray), 
        "reprojected_err" (float) and "im_filename" (str)

        NOTE: im_filenames and cams should be 1-to-1 correspondence.

        Parameters
        ----------
        im_filenames : Iterable[str]
            Image filenames name with format  
            <timestep>/<camera_id>.jpg.
        cams : Iterable[Camera]
            Cameras corresponding to images from im_filenames.
        aruco: str
            OpenCV arUco dictionary.
        marker_size: float
            Real size of arUco markers.
        corner_refine: str
            See OpenCV corner refinement options. 
        flags: str
            Method to solve PnP - See OpenCV.
        brightness: int
            Image brightness preprocessing.
        contrast: int
            Image contrast preprocessing.
        marker_ids: Iterable[str]
            Which marker IDs to detected.

        Returns
        -------
        output : dict
            Camera-marker edge dictionary. Keys are tuples
            (<camera_id>, <timestamp>_<marker_id>).
            Values are dicts with "pose" (SE3), "corners" (np.ndarray), 
            "reprojected_err" (float) and "im_filename" (str).
    """
    assert len(im_filenames) == len(cams)
    print("\nMarker detection")
    print("Received {} images.".format(len(im_filenames)))

    num_workers = mp.cpu_count()
    print("Started pool of {} workers.".format(num_workers))

    f = partial(target_dict["detector"],
                target_dict=target_dict,
                brightness=brightness,
                contrast=contrast,
                flags=flags)
        
    with mp.Pool(num_workers) as pool:
        out = pool.starmap(f, zip(im_filenames, cams))
    print("Merging dictionaries...")

    # Remove None detections
    out = [d for d in out if d is not None]
    print("Found markers in {} images".format(len(out)))

    # Merge dictionaries and eliminate detections of markers with wrong id
    if target_dict["detector"] == estimate_pose_aruco_worker:
        out = {k: v for d in out for k, v in d.items() if k[-1].split('_')[-1] in target_dict['marker_ids']}
    else:
        out = {k: v for d in out for k, v in d.items()}
        
    print("Finished: {} markers detected.".format(len(out)))
    return out