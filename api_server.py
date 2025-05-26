import os
import time
import asyncio
import uuid
import base64
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import airsim
import numpy as np
import cv2
import threading
from typing import Dict, Optional, List, Any
import uvicorn

# 导入我们现有的算法模块
from qwen_vl_client import QwenVLClient
from detection_utils import detect_target_with_qwen, match_features, find_closest_keypoint

# 创建FastAPI应用程序
app = FastAPI(title="UE PixStreaming API", description="API for controlling AirSim through UE PixStreaming")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 创建图像存储目录
IMAGES_DIR = "static/images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局状态
control_client = None  # 用于无人机控制的客户端
image_client = None    # 用于图像处理的客户端
status_client = None   # 用于状态监控的客户端
camera_name = "0"
target_position = None
is_api_control_enabled = False
current_mission_status = {
    "mission_active": False,
    "progress": 0,
    "success": False,
    "message": "",
    "position": {"x": 0.0, "y": 0.0, "z": 0.0}
}
mission_thread = None

# 连接到AirSim
@app.post("/api/connect")
async def connect_to_airsim():
    global control_client, image_client, status_client, is_api_control_enabled
    
    try:
        # 创建控制客户端
        control_client = airsim.MultirotorClient()
        control_client.confirmConnection()
        
        # 创建图像处理客户端
        image_client = airsim.MultirotorClient()
        image_client.confirmConnection()
        
        # 创建状态监控客户端
        status_client = airsim.MultirotorClient()
        status_client.confirmConnection()
        
        return JSONResponse(content={"success": True, "message": "Connected to AirSim with all required clients"})
    except Exception as e:
        # 确保在出错时清理资源
        control_client = None
        image_client = None
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
        control_client.simSetCameraFov(camera_name,30)
        
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
        # 确保API控制已启用
        if not is_api_control_enabled:
            control_client.enableApiControl(True)
            control_client.armDisarm(True)
            is_api_control_enabled = True
        
        # 执行起飞
        control_client.takeoffAsync().join()
        control_client.simSetCameraFov(camera_name,30)
        
        return JSONResponse(content={"success": True, "message": "Takeoff successful"})
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)})

# 降落
@app.post("/api/land")
async def land():
    global control_client, is_api_control_enabled
    
    if not control_client:
        return JSONResponse(content={"success": False, "message": "Not connected to AirSim"})
    
    try:
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
        # 捕获图像
        responses = image_client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
        ])
        
        if not responses:
            return JSONResponse(content={"success": False, "message": "Failed to capture image"})
        
        # 处理图像数据
        response = responses[0]
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = img.reshape(response.height, response.width, 3)
        
        # 生成唯一文件名
        filename = f"drone_image_{int(time.time())}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        # 保存图像
        cv2.imwrite(filepath, img)
        
        # 返回图像URL
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
        # 获取无人机位置
        state = status_client.simGetGroundTruthKinematics()
        
        # 构建返回数据
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
    x0 = 640  # 主点x坐标（假设图像宽度为1280）
    y0 = 480  # 主点y坐标（假设图像高度为960）
    f = 1280/(2 * np.tan(np.radians(fov / 2)))  # 焦距
    
    return x0, y0, f, xs, ys, zs, q

# 四元数转旋转矩阵函数
def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵"""
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

# 相机坐标转世界坐标函数
def camera_to_world(camera_coords, rotation_matrix, translation_vector):
    """
    将相机坐标转换为UE的世界坐标（左手坐标系适配）
    - Y_camera -> Y_world（主光轴一致）
    - X_camera -> Z_world
    - Z_camera -> X_world
    """
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
    
    # 使用视差计算深度(Z)
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

# 目标检测任务
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
        # 如果提供了目标类型，则使用该目标类型进行检测
        detection_result = detect_target_with_qwen(img1_path, target_type=target_type)
        
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
        # 创建可视化图像
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

# 执行绕飞任务
def execute_fly_around(target_world_coords):
    global control_client, current_mission_status
    
    try:
        # 更新任务状态
        current_mission_status["mission_active"] = True
        current_mission_status["progress"] = 1
        current_mission_status["success"] = False
        current_mission_status["message"] = "Starting fly-around mission"
        
        # 设置飞行参数
        center = np.array([[target_world_coords["x"]], [target_world_coords["y"]]])
        radius = 500  # 圆的半径（米）
        speed = 15     # 速度（米/秒）
        clock_wise = True  # 旋转方向
        target_point = center  # 无人机始终朝向圆心
        altitude = -20  # 飞行高度
        num_circles = 1  # 完成的圈数
        
        # 初始化位置历史
        pos_reserve = np.array([[0.], [0.], [altitude]])
        
        # 控制增益
        k_radial = 0.6  # 径向控制增益
        
        # 可视化目标圆
        circle_points = []
        for i in range(36):  # 用36个点绘制圆
            angle = i * (2 * np.pi / 36)
            circle_x = center[0, 0] + radius * np.cos(angle)
            circle_y = center[1, 0] + radius * np.sin(angle)
            circle_points.append(airsim.Vector3r(circle_x, circle_y, altitude))
        
        # 可视化圆和目标
        control_client.simPlotLineStrip(circle_points, color_rgba=[0.0, 1.0, 0.0, 1.0], thickness=5.0, is_persistent=True)
        control_client.simPlotPoints([airsim.Vector3r(float(target_point[0, 0]), float(target_point[1, 0]), altitude)], 
                               size=20.0, color_rgba=[1.0, 1.0, 0.0, 1.0], is_persistent=True)
        
        # 移动到圆边缘开始
        start_x = center[0, 0] + radius
        start_y = center[1, 0]
        control_client.moveToPositionAsync(start_x, start_y, altitude, 20).join()
        time.sleep(1)
        
        # 获取初始位置进行角度跟踪
        state = control_client.simGetGroundTruthKinematics()
        pos = np.array([[state.position.x_val], [state.position.y_val], [state.position.z_val]])
        dp = pos[0:2] - center
        
        # 记录起始角度
        start_angle = np.arctan2(dp[1, 0], dp[0, 0])
        prev_angle = start_angle
        angle_accumulated = 0.0
        
        # 主控制循环
        while True:
            # 检查任务是否被取消
            if not current_mission_status["mission_active"]:
                break
                
            # 获取当前无人机位置（使用control_client避免资源竞争）
            state = control_client.simGetGroundTruthKinematics()
            pos = np.array([[state.position.x_val], [state.position.y_val], [state.position.z_val]])
            
            # 更新位置信息
            current_mission_status["position"] = {
                "x": float(pos[0, 0]),
                "y": float(pos[1, 0]),
                "z": float(pos[2, 0])
            }
            
            # 计算从圆心到无人机的向量
            dp = pos[0:2] - center
            current_radius = np.linalg.norm(dp)
            
            # 计算当前角度
            current_angle = np.arctan2(dp[1, 0], dp[0, 0])
            
            # 计算角度变化（处理角度跳变）
            angle_diff = current_angle - prev_angle
            
            # 处理角度跳变（当从接近+π到-π或从接近-π到+π）
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # 更新累积角度（顺时针为负，逆时针为正）
            if clock_wise:
                angle_accumulated -= angle_diff  # 顺时针时角度减小
            else:
                angle_accumulated += angle_diff  # 逆时针时角度增加
            
            # 更新进度
            progress = min(abs(angle_accumulated) / (2 * np.pi * num_circles) * 100, 99)
            current_mission_status["progress"] = progress
            
            # 检查是否完成了所需圈数
            if abs(angle_accumulated) >= 2 * np.pi * num_circles:
                current_mission_status["progress"] = 100
                current_mission_status["success"] = True
                current_mission_status["message"] = f"Completed {num_circles} circle(s)"
                break
            
            # 归一化径向向量
            if current_radius > 0.1:  # 避免除零
                dp_normalized = dp / current_radius
            else:
                dp_normalized = np.array([[1.0], [0.0]])  # 如果在中心，则默认方向
            
            # 计算径向速度分量（用于半径控制）
            radius_error = current_radius - radius
            v_radial = -k_radial * radius_error * dp_normalized
            
            # 计算切向速度方向向量
            theta = np.arctan2(dp[1, 0], dp[0, 0])
            if clock_wise:
                theta += np.pi / 2
            else:
                theta -= np.pi / 2
            v_tangential = speed * np.array([[np.cos(theta)], [np.sin(theta)]])
            
            # 计算最终速度命令
            v_cmd = v_radial + v_tangential
            
            # 必要时限制速度
            v_cmd_magnitude = np.linalg.norm(v_cmd)
            if v_cmd_magnitude > speed:
                v_cmd = v_cmd * (speed / v_cmd_magnitude)
            
            # 计算所需偏航角以指向目标
            target_direction = target_point - pos[0:2]
            yaw = np.arctan2(target_direction[1, 0], target_direction[0, 0])
            
            # 应用速度命令和偏航控制
            control_client.moveByVelocityAsync(
                v_cmd[0, 0], v_cmd[1, 0], 0.005, 1, 
                airsim.DrivetrainType.MaxDegreeOfFreedom, 
                airsim.YawMode(is_rate=False, yaw_or_rate=np.degrees(yaw))
            )
            
            # 更新位置历史和前一角度
            pos_reserve = pos
            prev_angle = current_angle
            
            # 短暂休眠
            time.sleep(0.02)
        
        # 任务完成，返回原点
        if current_mission_status["success"]:
            control_client.moveToPositionAsync(0, 0, -5, 10).join()
        
        # 最后更新任务状态
        current_mission_status["mission_active"] = False
        
    except Exception as e:
        current_mission_status["mission_active"] = False
        current_mission_status["success"] = False
        current_mission_status["message"] = str(e)
        
        # 尝试安全返回
        try:
            control_client.moveToPositionAsync(0, 0, -5, 10).join()
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
        # 创建一个简单的灰色图像，大小为640x480，带有文本
        img = np.ones((480, 640, 3), dtype=np.uint8) * 100  # 灰色背景
        cv2.putText(img, "等待相机图像...", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(placeholder_path, img)
        print(f"Created placeholder image at {placeholder_path}")

if __name__ == "__main__":
    main()