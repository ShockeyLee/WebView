import os
import sys
import time
import subprocess
import webbrowser
import argparse
import threading
import signal
import psutil
import shutil

# 定义进程列表
processes = []

def start_process(command, cwd=None, shell=False):
    """启动一个子进程并返回进程对象"""
    try:
        if shell:
            process = subprocess.Popen(command, shell=True, cwd=cwd)
        else:
            process = subprocess.Popen(command, cwd=cwd, shell=shell)
        processes.append(process)
        return process
    except Exception as e:
        print(f"启动进程失败: {e}")
        return None

def start_qwen_vl_server():
    """启动Qwen-VL API服务器"""
    print("正在启动Qwen-VL API服务器...")
    return start_process(["python", "qwen-vl.py"])

def start_api_server():
    """启动Web API服务器"""
    print("正在启动增强版Web API服务器...")
    return start_process(["python", "api_server.py"])

def launch_browser(url, delay=2):
    """延迟启动浏览器"""
    time.sleep(delay)
    print(f"正在打开浏览器: {url}")
    webbrowser.open(url)

def check_ue_pixstreaming(host="localhost", port=80, max_attempts=10):
    """检查UE PixStreaming服务是否可用"""
    import socket
    
    print("正在检查UE PixStreaming服务...")
    
    for i in range(max_attempts):
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"UE PixStreaming服务已检测到在 {host}:{port}")
                return True
        except (socket.timeout, ConnectionRefusedError):
            print(f"尝试 {i+1}/{max_attempts}: UE PixStreaming服务未就绪...")
            time.sleep(2)
    
    print(f"警告: 在{max_attempts}次尝试后无法连接到UE PixStreaming服务")
    return False

def setup_web_frontend():
    """设置Web前端文件"""
    print("准备Web前端文件...")
    
    # 确保静态目录存在
    os.makedirs("static", exist_ok=True)
    os.makedirs("static/images", exist_ok=True)
    
    # 创建增强版Web界面内容
    enhanced_web_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能无人机任务控制系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            overflow: hidden;
        }
        
        .main-container {
            display: flex;
            height: 100vh;
        }
        
        /* 左侧视频流区域 */
        .video-section {
            flex: 7;
            background: #000;
            position: relative;
            overflow: hidden;
        }
        
        .video-container {
            width: 100%;
            height: 100%;
            position: relative;
        }
        
        #pixstream-frame {
            width: 100%;
            height: 100%;
            border: none;
        }
        
        /* 搜索路径覆盖层 */
        .search-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 10;
        }
        
        #search-canvas {
            width: 100%;
            height: 100%;
        }
        
        .video-overlay {
            position: absolute;
            top: 20px;
            left: 20px;
            right: 20px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            pointer-events: none;
            z-index: 20;
        }
        
        .status-card {
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            padding: 15px 20px;
            border-radius: 12px;
            color: white;
            pointer-events: auto;
            font-size: 14px;
        }
        
        /* 右侧控制面板 */
        .control-section {
            flex: 3;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            display: flex;
            flex-direction: column;
            border-left: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* 顶部导航标签 */
        .tab-navigation {
            display: flex;
            background: rgba(255, 255, 255, 0.1);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .tab-button {
            flex: 1;
            padding: 15px 10px;
            background: none;
            border: none;
            color: #666;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            font-size: 13px;
        }
        
        .tab-button.active {
            color: #2a5298;
            background: rgba(42, 82, 152, 0.1);
        }
        
        .tab-button.active::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: #2a5298;
            border-radius: 3px 3px 0 0;
        }
        
        /* 标签页内容 */
        .tab-content {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        /* 通用卡片样式 */
        .section-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(42, 82, 152, 0.1);
        }
        
        .section-title {
            font-size: 16px;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
        }
        
        /* 控制按钮 */
        .control-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .control-btn {
            padding: 12px 16px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 13px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #2a5298, #1e3c72);
            color: white;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #56CCF2, #2F80ED);
            color: white;
        }
        
        .btn-warning {
            background: linear-gradient(135deg, #FFB946, #FF6B6B);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #FF6B6B, #FF8E8E);
            color: white;
        }
        
        .control-btn:disabled {
            background: #f1f3f4;
            color: #9aa0a6;
            cursor: not-allowed;
        }
        
        .control-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }
        
        /* 智能对话 */
        .chat-container {
            height: 180px;
            display: flex;
            flex-direction: column;
            border: 2px solid #f1f3f4;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .chat-messages {
            flex: 1;
            padding: 12px;
            overflow-y: auto;
            background: #fafbfc;
            font-size: 13px;
        }
        
        .chat-input-container {
            padding: 12px;
            background: white;
            border-top: 1px solid #e1e5e9;
            display: flex;
            gap: 8px;
        }
        
        .chat-input {
            flex: 1;
            padding: 10px 14px;
            border: 1px solid #e1e5e9;
            border-radius: 20px;
            outline: none;
            font-size: 13px;
        }
        
        .send-button {
            padding: 10px 16px;
            background: #2a5298;
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 13px;
        }
        
        /* 其他样式 */
        .status-info {
            font-size: 12px;
            margin-top: 10px;
        }
        
        .progress-bar {
            width: 100%;
            height: 6px;
            background: #f1f3f4;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 8px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #2a5298, #1e3c72);
            width: 0%;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- 左侧视频流区域 -->
        <div class="video-section">
            <div class="video-container">
                <iframe id="pixstream-frame" src="http://localhost:80/" allowfullscreen></iframe>
                
                <!-- 搜索路径覆盖层 -->
                <div class="search-overlay">
                    <canvas id="search-canvas"></canvas>
                </div>
                
                <div class="video-overlay">
                    <div class="status-card">
                        <div>连接状态: <span id="connection-status">未连接</span></div>
                    </div>
                    
                    <div class="status-card">
                        <div>位置 X: <span id="pos-x">0.00</span></div>
                        <div>位置 Y: <span id="pos-y">0.00</span></div>
                        <div>位置 Z: <span id="pos-z">0.00</span></div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 右侧控制面板 -->
        <div class="control-section">
            <!-- 导航标签 -->
            <div class="tab-navigation">
                <button class="tab-button active" data-tab="planning">🎯 任务规划</button>
                <button class="tab-button" data-tab="execution">🚁 执行监控</button>
                <button class="tab-button" data-tab="logs">📋 系统日志</button>
            </div>
            
            <!-- 任务规划页面 -->
            <div class="tab-content active" id="planning-tab">
                <!-- 智能对话区域 -->
                <div class="section-card">
                    <div class="section-title">💬 智能指令对话</div>
                    <div class="chat-container">
                        <div class="chat-messages" id="chat-messages">
                            <div style="margin-bottom: 10px; color: #666;">
                                欢迎使用智能无人机系统！您可以输入指令，例如："在海域搜索航母目标"
                            </div>
                        </div>
                        <div class="chat-input-container">
                            <input type="text" class="chat-input" id="chat-input" placeholder="输入任务指令...">
                            <button class="send-button" id="send-button">发送</button>
                        </div>
                    </div>
                </div>
                
                <!-- 任务控制区域 -->
                <div class="section-card">
                    <div class="section-title">🎮 任务控制</div>
                    <div class="control-buttons">
                        <button class="control-btn btn-primary" id="connect-btn">🔗 连接系统</button>
                        <button class="control-btn btn-success" id="takeoff-btn" disabled>🚀 起飞</button>
                        <button class="control-btn btn-warning" id="search-btn" disabled>🔍 网格搜索</button>
                        <button class="control-btn btn-primary" id="detect-btn" disabled>🎯 精确定位</button>
                        <button class="control-btn btn-success" id="flyaround-btn" disabled>🔄 绕飞目标</button>
                        <button class="control-btn btn-danger" id="land-btn" disabled>🛬 降落</button>
                    </div>
                </div>
            </div>
            
            <!-- 执行监控页面 -->
            <div class="tab-content" id="execution-tab">
                <div class="section-card">
                    <div class="section-title">📊 任务进度</div>
                    <div>当前任务：<span id="current-task">待机中</span></div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill"></div>
                    </div>
                    <div class="status-info">
                        <div>搜索覆盖率: <span id="search-coverage">0%</span></div>
                        <div>目标类型: <span id="target-type">未检测</span></div>
                    </div>
                </div>
            </div>
            
            <!-- 系统日志页面 -->
            <div class="tab-content" id="logs-tab">
                <div class="section-card" style="height: calc(100vh - 200px);">
                    <div class="section-title">📋 系统日志</div>
                    <div id="logs-content" style="height: calc(100% - 60px); overflow-y: auto; background: #f8f9fa; padding: 10px; border-radius: 8px; font-family: monospace; font-size: 12px;">
                        <div>系统已启动</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // 全局变量
        const API_BASE_URL = 'http://localhost:5000/api';
        
        // 标签页切换
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', () => {
                document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                button.classList.add('active');
                document.getElementById(button.dataset.tab + '-tab').classList.add('active');
            });
        });
        
        // 聊天功能
        const chatInput = document.getElementById('chat-input');
        const sendButton = document.getElementById('send-button');
        const chatMessages = document.getElementById('chat-messages');
        
        function addMessage(text, isUser = true) {
            const messageDiv = document.createElement('div');
            messageDiv.style.marginBottom = '10px';
            messageDiv.style.color = isUser ? '#2a5298' : '#666';
            messageDiv.innerHTML = `<strong>${isUser ? '用户' : '系统'}:</strong> ${text}`;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function sendMessage() {
            const text = chatInput.value.trim();
            if (text) {
                addMessage(text, true);
                chatInput.value = '';
                processCommand(text);
            }
        }
        
        function processCommand(command) {
            if (command.includes('搜索')) {
                addMessage('已接收搜索指令，请点击"网格搜索"开始任务', false);
            } else if (command.includes('连接')) {
                connectToAirSim();
            } else {
                addMessage('指令已接收，正在分析...', false);
            }
        }
        
        sendButton.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        // 添加日志
        function addLog(message, level = 'info') {
            const logsContent = document.getElementById('logs-content');
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.innerHTML = `[${timestamp}] ${level.toUpperCase()}: ${message}`;
            logsContent.appendChild(logEntry);
            logsContent.scrollTop = logsContent.scrollHeight;
        }
        
        // 更新进度
        function updateProgress(percentage, task = '执行中') {
            document.getElementById('progress-fill').style.width = percentage + '%';
            document.getElementById('current-task').textContent = task;
        }
        
        // API调用函数
        async function apiCall(endpoint, method = 'POST', data = null) {
            try {
                const options = { method };
                if (data) {
                    options.headers = { 'Content-Type': 'application/json' };
                    options.body = JSON.stringify(data);
                }
                
                const response = await fetch(API_BASE_URL + endpoint, options);
                const result = await response.json();
                
                if (result.success) {
                    addLog(result.message || '操作成功', 'success');
                } else {
                    addLog(result.message || '操作失败', 'error');
                }
                
                return result;
            } catch (error) {
                addLog('请求失败: ' + error.message, 'error');
                return { success: false, message: error.message };
            }
        }
        
        // 连接到AirSim
        async function connectToAirSim() {
            addLog('正在连接到AirSim...', 'info');
            const result = await apiCall('/connect');
            
            if (result.success) {
                document.getElementById('connection-status').textContent = '已连接';
                document.getElementById('takeoff-btn').disabled = false;
                document.getElementById('connect-btn').disabled = true;
                startStatusPolling();
            }
        }
        
        // 起飞
        async function takeoff() {
            const result = await apiCall('/takeoff');
            if (result.success) {
                document.getElementById('takeoff-btn').disabled = true;
                document.getElementById('land-btn').disabled = false;
                document.getElementById('search-btn').disabled = false;
                document.getElementById('detect-btn').disabled = false;
            }
        }
        
        // 网格搜索
        async function startGridSearch() {
            const searchParams = {
                area: 1000,
                spacing: 100,
                altitude: 20,
                target_type: ""
            };
            
            document.getElementById('search-btn').disabled = true;
            const result = await apiCall('/start_grid_search', 'POST', searchParams);
            
            if (result.success) {
                updateProgress(10, '网格搜索中');
                pollSearchProgress();
            } else {
                document.getElementById('search-btn').disabled = false;
            }
        }
        
        // 监控搜索进度
        async function pollSearchProgress() {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(API_BASE_URL + '/search_status');
                    const data = await response.json();
                    
                    if (data.search_active) {
                        updateProgress(data.progress, '搜索中');
                        document.getElementById('search-coverage').textContent = data.coverage + '%';
                        
                        if (data.target_found) {
                            clearInterval(interval);
                            addLog('发现目标!', 'success');
                            document.getElementById('target-type').textContent = data.target_type;
                            document.getElementById('detect-btn').disabled = false;
                        }
                    } else {
                        clearInterval(interval);
                        document.getElementById('search-btn').disabled = false;
                        updateProgress(data.success ? 100 : 0, data.success ? '搜索完成' : '搜索失败');
                    }
                } catch (error) {
                    console.error('搜索进度轮询错误:', error);
                }
            }, 1000);
        }
        
        // 精确检测
        async function detectTarget() {
            document.getElementById('detect-btn').disabled = true;
            const result = await apiCall('/detect_target');
            
            if (result.success) {
                document.getElementById('flyaround-btn').disabled = false;
                document.getElementById('target-type').textContent = result.target_type;
            }
            
            document.getElementById('detect-btn').disabled = false;
        }
        
        // 绕飞目标
        async function flyAroundTarget() {
            document.getElementById('flyaround-btn').disabled = true;
            const result = await apiCall('/fly_around');
            
            if (result.success) {
                pollMissionProgress();
            } else {
                document.getElementById('flyaround-btn').disabled = false;
            }
        }
        
        // 监控绕飞进度
        async function pollMissionProgress() {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(API_BASE_URL + '/mission_status');
                    const data = await response.json();
                    
                    if (data.mission_active) {
                        updateProgress(data.progress, '绕飞中');
                    } else {
                        clearInterval(interval);
                        document.getElementById('flyaround-btn').disabled = false;
                        updateProgress(data.success ? 100 : 0, data.success ? '绕飞完成' : '绕飞失败');
                    }
                } catch (error) {
                    console.error('绕飞进度轮询错误:', error);
                }
            }, 500);
        }
        
        // 降落
        async function land() {
            const result = await apiCall('/land');
            if (result.success) {
                // 重置按钮状态
                document.getElementById('land-btn').disabled = true;
                document.getElementById('takeoff-btn').disabled = false;
                document.getElementById('search-btn').disabled = true;
                document.getElementById('detect-btn').disabled = true;
                document.getElementById('flyaround-btn').disabled = true;
            }
        }
        
        // 状态轮询
        function startStatusPolling() {
            setInterval(async () => {
                try {
                    const response = await fetch(API_BASE_URL + '/drone_status');
                    const data = await response.json();
                    
                    if (data.connected) {
                        document.getElementById('pos-x').textContent = data.position.x.toFixed(2);
                        document.getElementById('pos-y').textContent = data.position.y.toFixed(2);
                        document.getElementById('pos-z').textContent = data.position.z.toFixed(2);
                    }
                } catch (error) {
                    console.error('状态轮询错误:', error);
                }
            }, 1000);
        }
        
        // 事件监听器
        document.getElementById('connect-btn').addEventListener('click', connectToAirSim);
        document.getElementById('takeoff-btn').addEventListener('click', takeoff);
        document.getElementById('land-btn').addEventListener('click', land);
        document.getElementById('search-btn').addEventListener('click', startGridSearch);
        document.getElementById('detect-btn').addEventListener('click', detectTarget);
        document.getElementById('flyaround-btn').addEventListener('click', flyAroundTarget);
        
        // 初始化
        addLog('智能无人机系统已启动', 'success');
        addMessage('系统已就绪，请先连接AirSim开始任务', false);
    </script>
</body>
</html>'''
    
    # 保存增强版Web界面
    with open("web_pixstreaming_app.html", "w", encoding="utf-8") as f:
        f.write(enhanced_web_content)
    
    with open("static/index.html", "w", encoding="utf-8") as f:
        f.write(enhanced_web_content)
    
    print("Web前端文件准备完成")

def check_dependencies():
    """检查项目依赖"""
    print("检查项目依赖...")
    
    required_files = [
        "detection_utils.py",
        "qwen_vl_client.py", 
        "qwen-vl.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"警告: 缺少以下文件: {', '.join(missing_files)}")
        return False
    
    print("依赖检查完成")
    return True

def cleanup():
    """清理所有启动的进程"""
    print("\n正在关闭所有服务...")
    
    for process in processes:
        if process and process.poll() is None:
            try:
                # 获取进程的子进程
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                
                # 先尝试正常终止
                process.terminate()
                
                # 等待进程结束
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # 如果超时，强制终止
                    process.kill()
                
                # 终止所有子进程
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    
    print("所有服务已关闭")

def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    print("\n接收到中断信号，正在清理...")
    cleanup()
    sys.exit(0)

def print_system_info():
    """打印系统信息"""
    print("=" * 60)
    print("智能无人机任务控制系统 - 增强版")
    print("=" * 60)
    print("功能特性:")
    print("  ✓ 网格搜索目标检测")
    print("  ✓ 智能语音指令处理")
    print("  ✓ 实时搜索路径可视化")
    print("  ✓ 精确目标定位与绕飞")
    print("  ✓ 多页面管理界面")
    print("  ✓ 实时任务进度监控")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="启动增强版UE PixStreaming Web控制面板")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--port", type=int, default=5000, help="API服务器端口 (默认: 5000)")
    parser.add_argument("--skip-ue-check", action="store_true", help="跳过UE PixStreaming检查")
    parser.add_argument("--dev-mode", action="store_true", help="开发模式（显示详细日志）")
    args = parser.parse_args()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 打印系统信息
        print_system_info()
        
        # 检查项目依赖
        if not check_dependencies():
            print("项目依赖检查失败，但将继续启动...")
        
        # 设置Web前端
        # setup_web_frontend()
        
        # 检查UE PixStreaming服务
        if not args.skip_ue_check:
            ue_available = check_ue_pixstreaming()
            if not ue_available:
                print("\n警告: UE PixStreaming服务不可用!")
                print("请确保您的Unreal Engine项目正在运行并启用了PixStreaming功能")
                response = input("是否继续启动控制面板? (y/n): ")
                if response.lower() != 'y':
                    print("启动已取消")
                    return
        else:
            print("已跳过UE PixStreaming检查")
        
        # 启动Qwen-VL服务器
        print("\n启动服务组件...")
        qwen_process = start_qwen_vl_server()
        if not qwen_process:
            print("启动Qwen-VL服务器失败，但将继续...")
        else:
            print("✓ Qwen-VL服务器启动中...")
            time.sleep(3)
        
        # 启动增强版API服务器
        api_process = start_api_server()
        if not api_process:
            print("启动API服务器失败")
            return
        
        print("✓ API服务器启动中...")
        time.sleep(2)
        
        print("\n" + "=" * 60)
        print("🚀 所有服务已启动")
        print("=" * 60)
        print(f"📱 Web控制面板: http://localhost:{args.port}")
        print("🎮 UE PixStreaming: http://localhost:80")
        print("🤖 Qwen-VL API: http://localhost:8000")
        print("=" * 60)
        print("\n功能说明:")
        print("  1. 点击'连接系统'建立与AirSim的连接")
        print("  2. 使用智能对话输入自然语言指令")
        print("  3. 配置搜索参数后执行网格搜索任务")
        print("  4. 发现目标后进行精确定位和绕飞")
        print("  5. 实时监控任务进度和系统日志")
        print("\n按 Ctrl+C 关闭所有服务")
        print("=" * 60)
        
        # 打开浏览器
        if not args.no_browser:
            browser_thread = threading.Thread(
                target=launch_browser,
                args=(f"http://localhost:{args.port}",)
            )
            browser_thread.daemon = True
            browser_thread.start()
        
        # 监控服务状态
        while True:
            time.sleep(5)
            
            # 检查进程状态
            running_processes = 0
            for process in processes:
                if process and process.poll() is None:
                    running_processes += 1
            
            if args.dev_mode:
                print(f"运行中的服务: {running_processes}/{len(processes)}")
            
            # 如果所有进程都停止了，退出
            if running_processes == 0:
                print("所有服务进程已停止")
                break
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"启动过程中发生错误: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main()