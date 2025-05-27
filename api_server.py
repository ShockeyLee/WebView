# 使用Qwen-VLimport os
import time
import asyncio
import uuid
import base64
import math
import numpy as np
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import airsim
import cv2
import threading
from typing import Dict, Optional, List, Any
import uvicorn
import os
# 导入我们现有的算法模块
from qwen_vl_client import QwenVLClient
from detection_utils import detect_target_with_qwen, match_features, find_closest_keypoint

# 创建FastAPI应用程序
app = FastAPI(title="Enhanced UE PixStreaming API", description="Enhanced API for AirSim with grid search")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建图像存储目录
IMAGES_DIR = "static/images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局状态
control_client = None
image_client = None
status_client = None
camera_name = "0"
target_position = None
is_api_control_enabled = False
latest_qwen_result_image = None  # 存储最新的Qwen检测结果图像

# 任务状态管理
current_mission_status = {
    "mission_active": False,
    "progress": 0,
    "success": False,
    "message": "",
    "position": {"x": 0.0, "y": 0.0, "z": 0.0}
}

# 搜索状态管理
current_search_status = {
    "search_active": False,
    "progress": 0,
    "coverage": 0,
    "target_found": False,
    "target_type": None,
    "current_position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "current_index": 0,
    "total_points": 0,
    "success": False,
    "message": ""
}

mission_thread = None
search_thread = None

class SearchParams(BaseModel):
    area: int = 1000
    spacing: int = 100
    altitude: int = 20
    target_type: str = ""

# 连接到AirSim
@app.post("/api/connect")
async def connect_to_airsim():
    global control_client, image_client, status_client, is_api_control_enabled
    
    try:
        control_client = airsim.MultirotorClient()
        control_client.confirmConnection()
        
        image_client = airsim.MultirotorClient()
        image_client.confirmConnection()
        
        status_client = airsim.MultirotorClient()
        status_client.confirmConnection()
        
        return JSONResponse(content={"success": True, "message": "Connected to AirSim with all required clients"})
    except Exception as e:
        control_client = None
        image_client = None
        status_client = None
        return JSONResponse(content={"success": False, "message": str(e)})

# 启用API控制
@app.post("/api/enable_control")
async def enable_api_control():
    global control_client, is_api_control_enabled
    
    if not control_client:
        return JSONResponse(content={"success": False, "message": "Not connected to AirSim"})
    
    try:
        control_client.enableApiControl(True)
        control_client.armDisarm(True)
        is_api_control_enabled = True
        control_client.simSetCameraFov(camera_name, 30)
        
        return JSONResponse(content={"success": True, "message": "API control enabled"})
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)})

# 起飞
@app.post("/api/takeoff")
async def takeoff():
    global control_client, is_api_control_enabled
    
    if not control_client:
        return JSONResponse(content={"success": False, "message": "Not connected to AirSim"})
    
    try:
        if not is_api_control_enabled:
            control_client.enableApiControl(True)
            control_client.armDisarm(True)
            is_api_control_enabled = True
        
        control_client.takeoffAsync().join()
        control_client.simSetCameraFov(camera_name, 30)
        
        return JSONResponse(content={"success": True, "message": "Takeoff successful"})
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)})

# 降落
@app.post("/api/land")
async def land():
    global control_client, current_search_status
    
    if not control_client:
        return JSONResponse(content={"success": False, "message": "Not connected to AirSim"})
    
    try:
        # 停止所有正在进行的任务
        current_search_status["search_active"] = False
        current_mission_status["mission_active"] = False
        
        control_client.landAsync().join()
        
        return JSONResponse(content={"success": True, "message": "Landing successful"})
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)})

# 捕获图像
@app.post("/api/capture_image")
async def capture_image():
    global image_client, camera_name
    
    if not image_client:
        return JSONResponse(content={"success": False, "message": "Not connected to AirSim"})
    
    try:
        responses = image_client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
        ])
        
        if not responses:
            return JSONResponse(content={"success": False, "message": "Failed to capture image"})
        
        response = responses[0]
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = img.reshape(response.height, response.width, 3)
        
        filename = f"drone_image_{int(time.time())}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        cv2.imwrite(filepath, img)
        
        image_url = f"/static/images/{filename}"
        
        return JSONResponse(content={"success": True, "image_url": image_url})
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)})

# 获取无人机状态
@app.get("/api/drone_status")
async def get_drone_status():
    global status_client
    
    if not status_client:
        return JSONResponse(content={"connected": False})
    
    try:
        state = status_client.simGetGroundTruthKinematics()
        
        return JSONResponse(content={
            "connected": True,
            "position": {
                "x": state.position.x_val,
                "y": state.position.y_val,
                "z": state.position.z_val
            },
            "orientation": {
                "w": state.orientation.w_val,
                "x": state.orientation.x_val,
                "y": state.orientation.y_val,
                "z": state.orientation.z_val
            },
            "is_armed": status_client.isApiControlEnabled()
        })
    except Exception as e:
        return JSONResponse(content={"connected": False, "error": str(e)})

# 生成网格搜索路径
def generate_grid_search_path(area: int, spacing: int, altitude: int):
    """生成S型网格搜索路径"""
    search_path = []
    grid_points = int(area / spacing)
    start_x = -area / 2
    start_y = -area / 2
    
    # S型搜索路径
    for row in range(grid_points + 1):
        y = start_y + row * spacing
        
        if row % 2 == 0:
            # 从左到右
            for col in range(grid_points + 1):
                x = start_x + col * spacing
                search_path.append({"x": x, "y": y, "z": -altitude})
        else:
            # 从右到左
            for col in range(grid_points, -1, -1):
                x = start_x + col * spacing
                search_path.append({"x": x, "y": y, "z": -altitude})
    
    return search_path

# 执行网格搜索任务
def execute_grid_search(search_params: SearchParams):
    global control_client, image_client, current_search_status, target_position
    
    try:
        # 重置搜索状态
        current_search_status.update({
            "search_active": True,
            "progress": 0,
            "coverage": 0,
            "target_found": False,
            "target_type": None,
            "current_index": 0,
            "success": False,
            "message": "Starting grid search"
        })
        
        # 生成搜索路径
        search_path = generate_grid_search_path(
            search_params.area, 
            search_params.spacing, 
            search_params.altitude
        )
        
        current_search_status["total_points"] = len(search_path)
        
        # 执行搜索
        for i, point in enumerate(search_path):
            if not current_search_status["search_active"]:
                break
                
            # 更新当前位置和进度
            current_search_status["current_index"] = i
            current_search_status["current_position"] = point
            current_search_status["progress"] = int((i / len(search_path)) * 100)
            current_search_status["coverage"] = int((i / len(search_path)) * 100)
            
            # 移动到搜索点
            control_client.moveToPositionAsync(
                point["x"], point["y"], point["z"], 15
            ).join()
            
            # 等待稳定
            time.sleep(1)
            
            # 捕获图像并检测目标
            try:
                responses = image_client.simGetImages([
                    airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
                ])
                
                if responses:
                    response = responses[0]
                    img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img = img.reshape(response.height, response.width, 3)
                    
                    # 保存搜索图像
                    filename = f"search_image_{i}_{int(time.time())}.png"
                    filepath = os.path.join(IMAGES_DIR, filename)
                    cv2.imwrite(filepath, img)
                    
                    # 使用Qwen-VL检测目标
                    detection_result = detect_target_with_qwen(
                        filepath, 
                        target_type=search_params.target_type
                    )
                    
                    # 保存Qwen检测结果图像路径
                    global latest_qwen_result_image
                    latest_qwen_result_image = f"/static/images/{filename}"
                    
                    if detection_result and detection_result.get("bbox_2d"):
                        # 发现目标！
                        current_search_status.update({
                            "target_found": True,
                            "target_type": detection_result.get("sub_label") or detection_result.get("label"),
                            "success": True,
                            "message": f"Target found at position {i+1}/{len(search_path)}"
                        })
                        
                        # 保存目标位置（简化版，实际应该进行精确定位）
                        target_position = {
                            "x": float(point["x"]),
                            "y": float(point["y"]),
                            "z": float(point["z"])
                        }
                        
                        break
                        
            except Exception as e:
                print(f"Detection error at point {i}: {e}")
                continue
        
        # 搜索完成
        current_search_status["search_active"] = False
        if not current_search_status["target_found"]:
            current_search_status.update({
                "progress": 100,
                "coverage": 100,
                "success": True,
                "message": "Search completed - no target found"
            })
        
    except Exception as e:
        current_search_status.update({
            "search_active": False,
            "success": False,
            "message": str(e)
        })

# 开始网格搜索
@app.post("/api/start_grid_search")
async def start_grid_search(search_params: SearchParams):
    global control_client, search_thread, current_search_status
    
    if not control_client:
        return JSONResponse(content={"success": False, "message": "Not connected to AirSim"})
    
    # 如果已经有搜索任务在运行，先停止它
    if search_thread and search_thread.is_alive():
        current_search_status["search_active"] = False
        search_thread.join(timeout=2.0)
    
    # 启动新的搜索任务线程
    search_thread = threading.Thread(target=execute_grid_search, args=(search_params,))
    search_thread.daemon = True
    search_thread.start()
    
    return JSONResponse(content={"success": True, "message": "Grid search started"})

# 获取搜索状态
@app.get("/api/search_status")
async def get_search_status():
    global current_search_status
    return JSONResponse(content=current_search_status)

# 获取Qwen检测结果
@app.get("/api/get_qwen_result")
async def get_qwen_result():
    global latest_qwen_result_image
    
    if latest_qwen_result_image:
        return JSONResponse(content={
            "success": True, 
            "image_url": latest_qwen_result_image
        })
    else:
        return JSONResponse(content={
            "success": False, 
            "message": "No Qwen result available"
        })

# 获取相机参数
def get_camera_params(client, camera_name):
    """获取相机内外参数"""
    camera_info = client.simGetCameraInfo(camera_name)
    xs = camera_info.pose.position.x_val
    ys = camera_info.pose.position.y_val
    zs = camera_info.pose.position.z_val
    q = camera_info.pose.orientation
    fov = camera_info.fov
    
    # 相机内参
    x0 = 640  # 主点x坐标
    y0 = 480  # 主点y坐标
    f = 1280/(2 * np.tan(np.radians(fov / 2)))  # 焦距
    
    return x0, y0, f, xs, ys, zs, q

# 四元数转旋转矩阵
def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵"""
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

# 相机坐标转世界坐标
def camera_to_world(camera_coords, rotation_matrix, translation_vector):
    """将相机坐标转换为UE的世界坐标"""
    camera_coords_adjusted = np.array([
        camera_coords[2],  # Z_camera -> X_world
        camera_coords[0],  # X_camera -> Z_world
        camera_coords[1]   # Y_camera -> Y_world
    ])
    world_coords = np.dot(rotation_matrix, camera_coords_adjusted) + translation_vector
    return world_coords

# 基于视差计算深度和位置
def calculate_depth_and_position(pixel1, pixel2, camera_params1, camera_params2):
    """使用视差法计算目标的3D坐标"""
    x0_1, y0_1, f_1, xs_1, ys_1, zs_1, q1 = camera_params1
    x0_2, y0_2, f_2, xs_2, ys_2, zs_2, q2 = camera_params2
    
    x1, y1 = pixel1
    x2, y2 = pixel2
    
    # 计算视差
    disparity = abs(x1 - x2)
    
    if disparity == 0:
        raise ValueError("视差为零，无法计算深度")
    
    # 计算两相机之间的基线距离
    b = np.sqrt((xs_1-xs_2)**2+(ys_1-ys_2)**2)
    Z = f_1*b / disparity
    
    # 计算相机坐标系中的真实坐标
    X = (x1 - x0_1) * Z / f_1
    Y = (y1 - y0_1) * Z / f_1
    
    # 同时计算第二个相机的坐标（用于验证）
    X_ = (x2 - x0_2) * Z / f_2
    Y_ = (y2 - y0_2) * Z / f_2
    
    return X, Y, Z, X_, Y_

# 精确目标检测
@app.post("/api/detect_target")
async def detect_target(target_type: str = None):
    global control_client, image_client, camera_name, target_position
    
    if not control_client or not image_client:
        return JSONResponse(content={"success": False, "message": "Not connected to AirSim"})
    
    try:
        # 1. 移动到第一个位置
        control_client.moveToPositionAsync(35, 0, -15, 5).join()
        time.sleep(2)
        
        # 2. 在第一个位置拍照
        responses1 = image_client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
        ])
        
        img1 = np.frombuffer(responses1[0].image_data_uint8, dtype=np.uint8)
        img1 = img1.reshape(responses1[0].height, responses1[0].width, 3)
        
        # 保存第一张图像
        img1_filename = f"position1_{int(time.time())}.png"
        img1_path = os.path.join(IMAGES_DIR, img1_filename)
        cv2.imwrite(img1_path, img1)
        
        # 获取第一个相机参数
        camera_params1 = get_camera_params(image_client, camera_name)
        
        # 3. 移动到第二个位置
        control_client.moveByVelocityBodyFrameAsync(0, 10, 0, 10).join()
        control_client.moveByVelocityAsync(0, 0, 0, 1).join()
        time.sleep(2)
        
        # 4. 在第二个位置拍照
        responses2 = image_client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
        ])
        
        img2 = np.frombuffer(responses2[0].image_data_uint8, dtype=np.uint8)
        img2 = img2.reshape(responses2[0].height, responses2[0].width, 3)
        
        # 保存第二张图像
        img2_filename = f"position2_{int(time.time())}.png"
        img2_path = os.path.join(IMAGES_DIR, img2_filename)
        cv2.imwrite(img2_path, img2)
        
        # 获取第二个相机参数
        camera_params2 = get_camera_params(image_client, camera_name)
        
        # 5. 使用Qwen-VL模型检测目标
        detection_result = detect_target_with_qwen(img1_path, target_type=target_type)
        
        # 保存Qwen检测结果图像路径
        global latest_qwen_result_image
        latest_qwen_result_image = f"/static/images/{img1_filename}"
        
        if not detection_result or not detection_result["bbox_2d"]:
            return JSONResponse(content={
                "success": False, 
                "message": "No target detected",
                "image_url": f"/static/images/{img1_filename}"
            })
            
        # 获取检测框
        bbox = detection_result["bbox_2d"]
        
        # 6. 使用LightGlue和SuperPoint进行特征匹配
        kpts0, kpts1, m_kpts0, m_kpts1, matches = match_features(img1_path, img2_path)
        
        # 7. 找到距离检测框中心最近的特征点
        closest_idx = find_closest_keypoint(m_kpts0, bbox)
        
        if closest_idx is None:
            return JSONResponse(content={
                "success": False, 
                "message": "Failed to find matching feature points",
                "image_url": f"/static/images/{img1_filename}"
            })
            
        # 8. 使用视差法计算目标的3D位置
        pixel1 = (m_kpts0[closest_idx][0], m_kpts0[closest_idx][1])
        pixel2 = (m_kpts1[closest_idx][0], m_kpts1[closest_idx][1])
        
        # 计算视差值
        disparity = abs(pixel1[0] - pixel2[0])
        
        X, Y, Z, X_, Y_ = calculate_depth_and_position(pixel1, pixel2, camera_params1, camera_params2)
        
        # 9. 转换为世界坐标
        x0_1, y0_1, f_1, xs_1, ys_1, zs_1, q1 = camera_params1
        R1 = quaternion_to_rotation_matrix(q1)
        T1 = np.array([xs_1, ys_1, zs_1])
        target_world = camera_to_world(np.array([X, Y, Z]), R1, T1)
        
        # 保存目标位置
        target_position = {
            "x": float(target_world[0]),
            "y": float(target_world[1]),
            "z": float(target_world[2])
        }
        
        # 10. 可视化检测结果
        visualization_img = img1.copy()
        
        # 绘制检测框
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(visualization_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制特征点
        for kp in m_kpts0:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(visualization_img, (x, y), 3, (0, 255, 255), -1)
        
        # 特别标注最近的特征点
        x, y = int(m_kpts0[closest_idx][0]), int(m_kpts0[closest_idx][1])
        cv2.circle(visualization_img, (x, y), 5, (255, 0, 0), -1)
        
        # 绘制文本信息
        cv2.putText(visualization_img, f"Target: {detection_result.get('sub_label') or detection_result.get('label')}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(visualization_img, f"Keypoint: #{closest_idx} ({x}, {y})", 
                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(visualization_img, f"Disparity: {disparity:.2f} px", 
                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(visualization_img, f"Depth: {Z:.2f} m", 
                  (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 保存可视化结果
        vis_filename = f"detection_result_{int(time.time())}.png"
        vis_path = os.path.join(IMAGES_DIR, vis_filename)
        cv2.imwrite(vis_path, visualization_img)
        
        # 获取匹配信息数据
        match_info = {
            "match_count": len(m_kpts0),
            "selected_keypoint_id": closest_idx,
            "disparity": float(disparity),
            "depth": float(Z)
        }
        
        # 返回检测结果
        return JSONResponse(content={
            "success": True,
            "target_type": detection_result.get("sub_label") or detection_result.get("label"),
            "target_position": target_position,
            "match_info": match_info,
            "image_url": f"/static/images/{vis_filename}"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"success": False, "message": str(e)})

# PID控制器类
class PIDController:
    """简单的PID控制器"""
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()
    
    def update(self, current_value):
        current_time = time.time()
        dt = current_time - self.previous_time
        
        if dt <= 0.0:
            dt = 0.01
        
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.previous_error = error
        self.previous_time = current_time
        
        return output
    
    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()

# 执行绕飞任务 - 改进版
def execute_fly_around(target_world_coords):
    global control_client, current_mission_status
    
    try:
        # 更新任务状态
        current_mission_status["mission_active"] = True
        current_mission_status["progress"] = 1
        current_mission_status["success"] = False
        current_mission_status["message"] = "Starting fly-around mission"
        
        # 设置飞行参数
        center_x = target_world_coords["x"]
        center_y = target_world_coords["y"]
        altitude = -200.0  # 绕飞高度
        radius = 1000.0     # 绕飞半径
        speed = 8.0       # 绕飞速度
        clockwise = True  # 顺时针
        num_circles = 1   # 绕飞圈数
        
        print(f"开始圆周绕飞 - 中心:({center_x:.1f}, {center_y:.1f}) 高度:{altitude}m 半径:{radius}m")
        
        # 获取当前位置
        state = control_client.simGetGroundTruthKinematics()
        current_pos = np.array([state.position.x_val, state.position.y_val, state.position.z_val])
        
        # 计算最近的圆周点
        center_2d = np.array([center_x, center_y])
        current_2d = current_pos[0:2]
        
        # 从当前位置到圆心的向量
        to_center = center_2d - current_2d
        distance_to_center = np.linalg.norm(to_center)
        
        if distance_to_center > 0.1:  # 避免除零
            # 单位向量
            direction = to_center / distance_to_center
            # 圆周上最近的点
            nearest_point = center_2d - direction * radius
        else:
            # 如果当前就在圆心，默认选择圆心右侧的点
            nearest_point = np.array([center_x + radius, center_y])
        
        start_x = nearest_point[0]
        start_y = nearest_point[1]
        
        print(f"移动到圆周最近点: ({start_x:.1f}, {start_y:.1f}, {altitude:.1f})")
        
        # 初始化高度PID控制器
        altitude_pid = PIDController(kp=2.5, ki=0.15, kd=0.8, setpoint=altitude)
        
        # 控制参数
        k_radial = 0.8  # 径向控制增益
        max_v_z = 1.0   # 最大垂直速度
        
        center = np.array([[center_x], [center_y]])
        
        # 移动到起始点
        control_client.moveToPositionAsync(start_x, start_y, altitude, 15).join()
        time.sleep(1)
        
        # 获取实际位置并初始化角度跟踪
        state = control_client.simGetGroundTruthKinematics()
        actual_pos = np.array([state.position.x_val, state.position.y_val, state.position.z_val])
        pos = np.array([[actual_pos[0]], [actual_pos[1]], [actual_pos[2]]])
        
        dp = pos[0:2] - center
        prev_angle = math.atan2(dp[1, 0], dp[0, 0])
        angle_accumulated = 0.0
        
        print(f"开始绕飞，起始角度: {math.degrees(prev_angle):.2f}°")
        
        last_report_time = time.time()
        
        # 主控制循环
        while current_mission_status["mission_active"]:
            # 获取当前状态
            state = control_client.simGetGroundTruthKinematics()
            pos = np.array([[state.position.x_val], [state.position.y_val], [state.position.z_val]])
            current_altitude = pos[2, 0]
            
            # 更新位置信息
            current_mission_status["position"] = {
                "x": float(pos[0, 0]),
                "y": float(pos[1, 0]),
                "z": float(pos[2, 0])
            }
            
            # 计算径向信息
            dp = pos[0:2] - center
            current_radius = np.linalg.norm(dp)
            current_angle = math.atan2(dp[1, 0], dp[0, 0])
            
            # 计算角度变化（处理角度跳变）
            angle_diff = current_angle - prev_angle
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # 更新累积角度
            if clockwise:
                angle_accumulated -= angle_diff
            else:
                angle_accumulated += angle_diff
            
            # 更新进度
            progress = min(abs(angle_accumulated) / (2 * math.pi * num_circles) * 100, 99)
            current_mission_status["progress"] = progress
            
            # 状态报告（每2秒）
            current_time = time.time()
            if current_time - last_report_time >= 2.0:
                altitude_error = abs(current_altitude - altitude)
                radius_error = abs(current_radius - radius)
                print(f"进度:{progress:.1f}% 高度误差:{altitude_error:.2f}m 半径误差:{radius_error:.1f}m")
                last_report_time = current_time
            
            # 检查是否完成
            if abs(angle_accumulated) >= 2 * math.pi * num_circles:
                circles_completed = abs(angle_accumulated) / (2 * math.pi)
                print(f"✓ 绕飞完成! 实际完成:{circles_completed:.2f}圈")
                current_mission_status["progress"] = 100
                current_mission_status["success"] = True
                current_mission_status["message"] = f"Completed {circles_completed:.2f} circle(s)"
                break
            
            # 计算径向控制（保持半径）
            if current_radius > 0.1:
                dp_normalized = dp / current_radius
            else:
                dp_normalized = np.array([[1.0], [0.0]])
            
            radius_error = current_radius - radius
            v_radial = -k_radial * radius_error * dp_normalized
            
            # 计算切向速度（圆周运动）
            theta = math.atan2(dp[1, 0], dp[0, 0])
            if clockwise:
                theta += math.pi / 2
            else:
                theta -= math.pi / 2
            
            v_tangential = speed * np.array([[math.cos(theta)], [math.sin(theta)]])
            
            # 合成水平速度
            v_cmd_xy = v_radial + v_tangential
            
            # 限制水平速度
            v_cmd_xy_magnitude = np.linalg.norm(v_cmd_xy)
            if v_cmd_xy_magnitude > speed * 1.2:
                v_cmd_xy = v_cmd_xy * (speed * 1.2 / v_cmd_xy_magnitude)
            
            # PID高度控制
            v_z = altitude_pid.update(current_altitude)
            v_z = max(-max_v_z, min(max_v_z, v_z))
            
            # 计算偏航角（朝向圆心）
            target_direction = center - pos[0:2]
            yaw = math.atan2(target_direction[1, 0], target_direction[0, 0])
            
            # 发送控制命令
            control_client.moveByVelocityAsync(
                v_cmd_xy[0, 0], v_cmd_xy[1, 0], v_z, 0.05,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(yaw))
            )
            
            prev_angle = current_angle
            time.sleep(0.05)  # 20Hz控制频率
        
        # 绕飞完成后悬停
        print("绕飞结束，开始悬停...")
        control_client.hoverAsync().join()
        time.sleep(1)
        
        # 最后更新任务状态
        current_mission_status["mission_active"] = False
        
    except Exception as e:
        current_mission_status["mission_active"] = False
        current_mission_status["success"] = False
        current_mission_status["message"] = str(e)
        print(f"绕飞任务错误: {e}")
        
        # 尝试安全返回
        try:
            control_client.hoverAsync().join()
        except:
            pass

# 开始绕飞任务
@app.post("/api/fly_around")
async def start_fly_around():
    global target_position, control_client, mission_thread, current_mission_status
    
    if not control_client:
        return JSONResponse(content={"success": False, "message": "Not connected to AirSim"})
    
    if not target_position:
        return JSONResponse(content={"success": False, "message": "No target position available"})
    
    # 如果已经有任务在运行，先停止它
    if mission_thread and mission_thread.is_alive():
        current_mission_status["mission_active"] = False
        mission_thread.join(timeout=2.0)
    
    # 重置任务状态
    current_mission_status = {
        "mission_active": True,
        "progress": 0,
        "success": False,
        "message": "Starting mission",
        "position": {"x": 0.0, "y": 0.0, "z": 0.0}
    }
    
    # 启动新的任务线程
    mission_thread = threading.Thread(target=execute_fly_around, args=(target_position,))
    mission_thread.daemon = True
    mission_thread.start()
    
    return JSONResponse(content={"success": True, "message": "Mission started"})

# 获取任务状态
@app.get("/api/mission_status")
async def get_mission_status():
    global current_mission_status
    return JSONResponse(content=current_mission_status)

# Web应用主页
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

# 主函数
def main():
    # 确保静态目录存在
    os.makedirs("static", exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # 创建一个占位图像
    create_placeholder_image()
    
    # 将Web应用程序复制到静态目录
    with open("static/index.html", "w", encoding="utf-8") as f:
        with open("web_pixstreaming_app.html", "r", encoding="utf-8") as src:
            f.write(src.read())
    
    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=5000)

def create_placeholder_image():
    """创建一个占位图像"""
    placeholder_path = os.path.join(IMAGES_DIR, "placeholder.png")
    if not os.path.exists(placeholder_path):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 100
        cv2.putText(img, "等待相机图像...", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(placeholder_path, img)
        print(f"Created placeholder image at {placeholder_path}")

if __name__ == "__main__":
    main()