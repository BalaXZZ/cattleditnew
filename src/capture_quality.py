"""Conservative recapture flags, not a learned quality or pose certification."""
import cv2
import numpy as np
from PIL import Image

def capture_metrics(image):
    rgb=image.convert('RGB');w,h=rgb.size
    preview=rgb.resize((256,256),Image.Resampling.BILINEAR)
    gray=cv2.cvtColor(np.asarray(preview),cv2.COLOR_RGB2GRAY)
    return {'width':w,'height':h,'short_side':min(w,h),
            'sharpness':float(cv2.Laplacian(gray,cv2.CV_32F).var()),
            'contrast':float(np.percentile(gray,90)-np.percentile(gray,10)),
            'mean_luminance':float(gray.mean())}

def inspect_capture(image,policy):
    metrics=capture_metrics(image);flags=[]
    if metrics['short_side']<policy['min_short_side']:flags.append('muzzle_resolution_low')
    if metrics['sharpness']<policy['sharpness_review_below']:flags.append('possible_blur_or_low_texture')
    if metrics['contrast']<policy['contrast_review_below']:flags.append('low_muzzle_contrast')
    return {'metrics':metrics,'flags':flags,'note':'Heuristic review flags; frontal pose, occlusion and fingerprint visibility still require visual checking.'}
