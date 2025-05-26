import numpy as np
import cv2
import math
import torch
import time
import json
import os
import requests
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import matplotlib.pyplot as plt
from qwen_vl_client import QwenVLClient

# Initialize Qwen-VL client
qwen_client = QwenVLClient()

def detect_target_with_qwen(image_path, target_type=None):
    """
    Detect target in image using Qwen-VL model
    
    Args:
        image_path (str): Path to the image file
        target_type (str, optional): Type of target to detect
        
    Returns:
        dict: Detection result with bbox and labels or None if detection failed
    """
    # 根据目标类型生成提示
    if target_type:
        prompt = f"""Detect the {target_type} in the image and return its position in the form of coordinates. 
                  The output format should be {{"bbox_2d": [x1, y1, x2, y2], "label": "object", "sub_label": "{target_type}" }}. 
                  If you can't detect such a target, return {{"bbox_2d": [], "label": None, "sub_label": None }}. 
                  And briefly describe your thinking and reasoning process after the test results under the heading 
                  "**Reasoning Process:**", explaining your detection logic step by step."""
    else:
        # 默认检测航母
        prompt = """Detect the aircraft carrier in the image and return its position in the form of coordinates. 
                  The output format should be {"bbox_2d": [x1, y1, x2, y2], "label": "boat", "sub_label": "aircraft carrier" }. 
                  If you can't detect such a target, return {"bbox_2d": [], "label": None, "sub_label": None }. 
                  And briefly describe your thinking and reasoning process after the test results under the heading 
                  "**Reasoning Process:**", explaining your detection logic step by step."""
    
    # 发送请求到Qwen-VL API
    response = qwen_client.analyze_image(
        image_path=image_path,
        prompt=prompt,
        temperature=0.7,
        max_tokens=512
    )
    
    if not response:
        print("API request failed")
        return None
    
    # 提取JSON数据
    result = qwen_client.extract_json_from_response(response)
    
    if not result:
        print("Failed to extract detection results")
        return None
    
    # 检查是否检测到目标
    if not result.get("bbox_2d"):
        print("No target detected")
        return result
    
    print(f"Target detected: {result.get('label')} - {result.get('sub_label')}")
    return result

def match_features(image1_path, image2_path):
    """
    Match features between two images using LightGlue and SuperPoint
    
    Args:
        image1_path (str): Path to first image
        image2_path (str): Path to second image
        
    Returns:
        tuple: (keypoints1, keypoints2, matched_keypoints1, matched_keypoints2, matches)
    """
    # Check if CUDA is available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load feature extractor and matcher
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    
    # Load images
    image0 = load_image(image1_path).to(device)
    image1 = load_image(image2_path).to(device)
    
    # Extract local features
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    
    # Match features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # Remove batch dimension
    
    # Get match results
    kpts0, kpts1 = feats0['keypoints'], feats1['keypoints']
    matches = matches01['matches']
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]  # Matched keypoints
    
    # Convert keypoints to numpy arrays for further processing
    kpts0_np = kpts0.cpu().numpy()
    kpts1_np = kpts1.cpu().numpy()
    m_kpts0_np = m_kpts0.cpu().numpy()
    m_kpts1_np = m_kpts1.cpu().numpy()
    matches_np = matches.cpu().numpy()
    
    return kpts0_np, kpts1_np, m_kpts0_np, m_kpts1_np, matches_np

def find_closest_keypoint(keypoints, bbox):
    """
    Find the keypoint closest to the bbox center
    
    Args:
        keypoints (numpy.ndarray): Array of keypoints
        bbox (list): Bounding box [x1, y1, x2, y2]
        
    Returns:
        int: Index of the closest keypoint or None if bbox is empty
    """
    if not bbox:
        return None
    
    # Calculate bbox center
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Find the closest keypoint
    min_dist = float('inf')
    closest_idx = -1
    
    for i, kp in enumerate(keypoints):
        x, y = kp
        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    return closest_idx

def visualize_matches(image1_path, image2_path, m_kpts0, m_kpts1, bbox=None, closest_idx=None, output_path="matches_visualization.png"):
    """
    Visualize feature matches and detection bbox
    
    Args:
        image1_path (str): Path to first image
        image2_path (str): Path to second image
        m_kpts0 (numpy.ndarray): Matched keypoints in first image
        m_kpts1 (numpy.ndarray): Matched keypoints in second image
        bbox (list, optional): Bounding box [x1, y1, x2, y2]
        closest_idx (int, optional): Index of closest keypoint to highlight
        output_path (str): Path to save visualization
        
    Returns:
        str: Path to saved visualization
    """
    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(15, 7))
    
    # Show first image
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.scatter(m_kpts0[:, 0], m_kpts0[:, 1], c='lime', s=5)
    
    # If bbox provided, draw it on first image
    if bbox:
        x1, y1, x2, y2 = bbox
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=2)
        # Draw bbox center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        plt.plot(center_x, center_y, 'ro', markersize=8)
    
    # If closest keypoint provided, highlight it
    if closest_idx is not None:
        plt.plot(m_kpts0[closest_idx, 0], m_kpts0[closest_idx, 1], 'bo', markersize=8)
    
    plt.title('Image 1 - Features and Detection')
    
    # Show second image
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.scatter(m_kpts1[:, 0], m_kpts1[:, 1], c='lime', s=5)
    
    # If closest keypoint provided, highlight it
    if closest_idx is not None:
        plt.plot(m_kpts1[closest_idx, 0], m_kpts1[closest_idx, 1], 'bo', markersize=8)
    
    plt.title('Image 2 - Matched Features')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path