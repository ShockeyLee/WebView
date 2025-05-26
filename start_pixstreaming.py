import os
import sys
import time
import subprocess
import webbrowser
import argparse
import threading
import signal
import psutil

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
    print("正在启动Web API服务器...")
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

def cleanup():
    """清理所有启动的进程"""
    print("\n正在关闭所有服务...")
    
    for process in processes:
        if process and process.poll() is None:  # 如果进程仍在运行
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

def main():
    parser = argparse.ArgumentParser(description="启动UE PixStreaming Web控制面板")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--port", type=int, default=5000, help="API服务器端口 (默认: 5000)")
    args = parser.parse_args()
    
    # 注册信号处理器，以便在按下Ctrl+C时正常退出
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 检查UE PixStreaming服务是否可用
        ue_available = check_ue_pixstreaming()
        if not ue_available:
            print("\n警告: UE PixStreaming服务不可用!")
            print("请确保您的Unreal Engine项目正在运行并启用了PixStreaming功能")
            response = input("是否继续启动控制面板? (y/n): ")
            if response.lower() != 'y':
                print("启动已取消")
                return
        
        # 确保静态目录存在
        os.makedirs("static", exist_ok=True)
        
        # 复制Web应用文件
        print("准备Web前端文件...")
        web_file_path = "web_pixstreaming_app.html"
        # 如果web_pixstreaming_app.html不存在，则使用complete_web_frontend.html
        if not os.path.exists(web_file_path) and os.path.exists("complete_web_frontend.html"):
            print("使用complete_web_frontend.html作为Web前端")
            web_file_path = "complete_web_frontend.html"
            
        with open(web_file_path, "r", encoding="utf-8") as f:
            web_content = f.read()
        
        # 保存到web_pixstreaming_app.html，这样API服务器可以找到它
        with open("web_pixstreaming_app.html", "w", encoding="utf-8") as f:
            f.write(web_content)
        
        # 启动QWEN-VL服务器
        qwen_process = start_qwen_vl_server()
        if not qwen_process:
            print("启动Qwen-VL服务器失败")
            return
        
        print("等待Qwen-VL服务器启动...")
        time.sleep(3)
        
        # 启动API服务器
        api_process = start_api_server()
        if not api_process:
            print("启动API服务器失败")
            return
        
        print("等待API服务器启动...")
        time.sleep(2)
        
        print("\n=== 所有服务已启动 ===")
        print(f"- Web控制面板: http://localhost:{args.port}")
        print("- UE PixStreaming: http://localhost:80")
        print("- Qwen-VL API: http://localhost:8000")
        print("\n按 Ctrl+C 关闭所有服务")
        
        # 打开浏览器
        if not args.no_browser:
            browser_thread = threading.Thread(
                target=launch_browser,
                args=(f"http://localhost:{args.port}",)
            )
            browser_thread.daemon = True
            browser_thread.start()
        
        # 等待子进程结束
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()