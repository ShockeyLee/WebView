'''
航母size: 332*89*77 BPA_WEST_AIRCRAFT_CARRIER
this script is used to generate a 3D bounding box for a specific object in the scene.
输入：场景中的物体名称 size
输出：物体的三维包围盒的像素坐标
'''

import os
import json
import cv2
import numpy as np
import airsim

# 获取场景物体的三维真实坐标和朝向
def get_object_position(client, object_name):
    object_state = client.simGetObjectPose(object_name)
    return object_state.position, object_state.orientation

def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵"""
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

# 将三维坐标转换为像素坐标
def world_to_pixel(client, position, camera_name='0'):
    camera_info = client.simGetCameraInfo(camera_name)
    fov = camera_info.fov
    width = 1280
    height = 960
    f = width / (2 * np.tan(fov * np.pi / 360))
    cx = width / 2
    cy = height / 2
    position = np.array([position[0], position[1], position[2]])

    # # 转换为相机坐标系
    camera_pose = camera_info.pose
    camera_translation = np.array([
        camera_pose.position.x_val,
        camera_pose.position.y_val,
        camera_pose.position.z_val
    ])
    rotation_matrix = quaternion_to_rotation_matrix(camera_pose.orientation)
    camera_position = np.linalg.inv(rotation_matrix) @ np.array(position-camera_translation)
    position = [camera_position[1], camera_position[2], camera_position[0]]
    # 转换为像素坐标
    u = int(f * position[0] / position[2] + cx)
    v = int(f * position[1] / position[2] + cy)

    return u, v

# 构建物体的三维包围盒
def build_object_bbox(client, object_name,size):
    position, orientation = get_object_position(client, object_name)
    # 假设物体的尺寸是固定的，这里可以根据实际情况进行调整
    
    # 计算物体的8个顶点
    vertices = np.array([
        [-size[0]/2, -size[1]/2, -size[2]/2],
        [size[0]/2, -size[1]/2, -size[2]/2],
        [size[0]/2, size[1]/2, -size[2]/2],
        [-size[0]/2, size[1]/2, -size[2]/2],
        [-size[0]/2, -size[1]/2, size[2]/2],
        [size[0]/2, -size[1]/2, size[2]/2],
        [size[0]/2, size[1]/2, size[2]/2],
        [-size[0]/2, size[1]/2, size[2]/2]
    ])
    # 旋转物体的顶点
    rotation_matrix = quaternion_to_rotation_matrix(orientation)

    vertices = np.dot(rotation_matrix, vertices.T).T
    # 平移物体的顶点
    vertices += np.array([
        position.x_val,
        position.y_val,
        position.z_val
    ])
    
    print("顶点世界坐标：",vertices)
    # 将顶点转换为像素坐标
    pixel_vertices = np.array([world_to_pixel(client, v) for v in vertices])
    print("顶点像素坐标",pixel_vertices)
    return pixel_vertices

# 绘制物体的三维包围盒
def draw_object_bbox(client, image, object_name,size):
    bbox = build_object_bbox(client, object_name,size)
    # 创建图像副本以确保可写
    image_copy = image.copy()
    # 绘制包围盒的8条边
    for i in range(4):
        cv2.line(image_copy, tuple(bbox[i]), tuple(bbox[(i+1)%4]), (0, 255, 0), 2)
        cv2.line(image_copy, tuple(bbox[i+4]), tuple(bbox[(i+1)%4+4]), (0, 255, 0), 2)
        cv2.line(image_copy, tuple(bbox[i]), tuple(bbox[i+4]), (0, 255, 0), 2)
    return image_copy

if __name__ == '__main__':
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    size_carrier = np.array([332, 89, 77])
    # size = np.array([50, 50, 30])
    size_Sovremenny = np.array([165, 22, 40])

    # 起飞
    client.takeoffAsync().join()
    client.simSetCameraFov('0',30)
    # 拍摄图片
    responses = client.simGetImages([
            airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)
        ])
        
    response = responses[0]
    img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img = img.reshape(response.height, response.width, 3)
    
    cv2.imwrite('test.png', img)
    # 绘制包围盒 ABP_West_Ship_DDG102_4  ABP_East_Ship_Sovremenny_4
    image = draw_object_bbox(client, img, 'BPA_West_Carrier_CVN76_9',size_carrier)
    image = draw_object_bbox(client, image, 'ABP_East_Ship_Sovremenny_4',size_Sovremenny)
    # image = draw_object_bbox(client, img, 'Cone2_5')
    cv2.imwrite('test_bbox.png', image)