from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
import requests
import json

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 替换为你的API密钥（建议通过环境变量获取）
API_KEY = "sk-uasenxkyttworvfdtosugkoxisvtyyrovvenzofpsweijzfq"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

@app.post("/generate")
async def generate_response(
    image: UploadFile = File(...),
    prompt: str = Form(None),  # 允许纯文本提示可选
    temperature: float = Form(0.7),
    max_tokens: int = Form(512)
):
    """主处理端点"""
    try:
        # 直接在这个函数中处理图片，避免数据传递问题
        image_data = await image.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")
        
        # 构建完整的data URI
        image_mime = image.content_type or "image/jpeg"  # 默认为jpeg如果没有指定
        data_uri = f"data:{image_mime};base64,{base64_image}"
        
        # 构建消息内容
        content = []
        
        # 添加图像内容
        content.append({
            "type": "image_url",
            "image_url": {
                "url": data_uri
            }
        })
        
        # 如果有文本提示则添加
        if prompt:
            content.append({"type": "text", "text": prompt})
        else:
            # 如果没有提示，添加一个默认提示
            content.append({"type": "text", "text": "请分析这张图片"})
        
        # 构造请求体
        payload = {
            "model": "Pro/Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [{
                "role": "user",
                "content": content
            }],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # 发送请求
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 打印请求信息以便调试，但不打印完整的base64数据
        print(f"Sending request to {API_URL}")
        print(f"Headers: {headers}")
        print(f"Content type: {image_mime}")
        print(f"Base64 string starts with: {base64_image[:30]}...")
        print(f"Data URI starts with: {data_uri[:50]}...")
        
        response = requests.post(API_URL, json=payload, headers=headers)
        
        # 打印响应状态和内容
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text[:200]}...")
        
        response.raise_for_status()
        
        return response.json()
    
    except requests.RequestException as e:
        error_detail = str(e)
        # 尝试获取更详细的错误信息
        if hasattr(e, 'response') and e.response:
            try:
                error_detail = f"{error_detail} - {e.response.text}"
            except:
                pass
        return {"error": f"API请求失败: {error_detail}", "status_code": e.response.status_code if hasattr(e, 'response') else None}
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)