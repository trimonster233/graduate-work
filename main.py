from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import os
from typing import List, Dict, Any
import uvicorn
import zipfile
from io import BytesIO
import asyncio
import json
import time
from contextlib import asynccontextmanager

# 存储活跃的连接
active_connections: Dict[str, Any] = {}

# 清理任务
cleanup_task = None

# 使用生命周期管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    global cleanup_task
    cleanup_task = asyncio.create_task(cleanup_expired_sessions())
    yield
    # 关闭时执行
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

app = FastAPI(title="视频文件发送服务", lifespan=lifespan)

# 视频文件目录
VIDEO_DIR = "./results/v15/avatars/0/"

# 确保目录存在
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR, exist_ok=True)

# 获取所有MP4文件并按名称排序
def get_sorted_videos() -> List[str]:
    if not os.path.exists(VIDEO_DIR):
        return []
    
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    videos.sort()  # 按文件名排序
    return videos

# 清理过期的会话
async def cleanup_expired_sessions():
    """定期清理过期的会话"""
    try:
        while True:
            current_time = time.time()
            expired_sessions = []
            
            # 找出过期的会话（30分钟未活动）
            for client_id, session in active_connections.items():
                if current_time - session["last_access"] > 1800:  # 30分钟
                    expired_sessions.append(client_id)
            
            # 删除过期会话
            for client_id in expired_sessions:
                del active_connections[client_id]
            
            # 等待10分钟再次检查
            await asyncio.sleep(600)
    except asyncio.CancelledError:
        print("清理任务已取消")

@app.get("/")
def read_root():
    return {"message": "视频文件发送服务已启动"}

@app.get("/videos", response_model=List[str])
def list_videos():
    """获取所有可用的视频文件列表"""
    videos = get_sorted_videos()
    if not videos:
        raise HTTPException(status_code=404, detail="未找到视频文件")
    return videos

@app.get("/videos/{video_id}")
async def get_video(video_id: int):
    """根据索引获取特定的视频文件"""
    videos = get_sorted_videos()
    if not videos:
        raise HTTPException(status_code=404, detail="未找到视频文件")
    
    if video_id < 0 or video_id >= len(videos):
        raise HTTPException(status_code=404, detail=f"视频索引 {video_id} 超出范围")
    
    video_path = os.path.join(VIDEO_DIR, videos[video_id])
    return FileResponse(
        path=video_path, 
        media_type="video/mp4", 
        filename=videos[video_id]
    )

@app.get("/videos/name/{video_name}")
async def get_video_by_name(video_name: str):
    """根据文件名获取特定的视频文件"""
    if not video_name.endswith(".mp4"):
        video_name += ".mp4"
    
    video_path = os.path.join(VIDEO_DIR, video_name)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"视频 {video_name} 不存在")
    
    return FileResponse(
        path=video_path, 
        media_type="video/mp4", 
        filename=video_name
    )



@app.get("/stream-videos")
async def stream_videos():
    """流式传输所有视频文件，每隔1秒发送一个视频的信息"""
    videos = get_sorted_videos()
    if not videos:
        raise HTTPException(status_code=404, detail="未找到视频文件")
    
    async def video_generator():
        for i, video in enumerate(videos):
            video_path = os.path.join(VIDEO_DIR, video)
            if os.path.exists(video_path):
                # 获取视频文件信息
                file_size = os.path.getsize(video_path)
                
                # 构建消息
                message = {
                    "index": i,
                    "filename": video,
                    "size": file_size,
                    "url": f"/videos/name/{video}",
                    "total_videos": len(videos),
                    "is_last": i == len(videos) - 1
                }
                
                # 发送消息
                yield f"data: {json.dumps(message)}\n\n"
                
                # 等待1秒
                await asyncio.sleep(1)
        
        # 发送结束消息
        end_message = {"status": "completed", "total_sent": len(videos)}
        yield f"data: {json.dumps(end_message)}\n\n"
    
    return StreamingResponse(
        video_generator(),
        media_type="text/event-stream"
    )

@app.post("/api/start-stream")
async def start_stream(request: Request):
    """
    启动一个视频流会话，返回会话ID
    客户端可以使用这个会话ID查询后续视频
    """
    client_id = f"client_{int(time.time())}_{len(active_connections) + 1}"
    
    # 获取视频列表
    videos = get_sorted_videos()
    if not videos:
        raise HTTPException(status_code=404, detail="未找到视频文件")
    
    # 创建会话信息
    active_connections[client_id] = {
        "videos": videos,
        "current_index": 0,
        "total": len(videos),
        "last_access": time.time(),
        "completed": False
    }
    
    # 返回会话信息
    return {
        "client_id": client_id,
        "total_videos": len(videos),
        "message": "视频流已启动，请使用获取的client_id查询视频"
    }

@app.get("/api/next-video/{client_id}")
async def get_next_video(client_id: str):
    """
    获取下一个视频信息
    如果没有更多视频，将返回完成状态
    """
    # 检查会话是否存在
    if client_id not in active_connections:
        raise HTTPException(status_code=404, detail="无效的会话ID，请重新开始")
    
    session = active_connections[client_id]
    session["last_access"] = time.time()
    
    # 如果已完成，返回状态
    if session["completed"]:
        return {
            "status": "completed",
            "total_videos": session["total"],
            "message": "所有视频已传输完成"
        }
    
    # 获取当前索引
    current_index = session["current_index"]
    
    # 检查是否已到末尾
    if current_index >= session["total"]:
        session["completed"] = True
        return {
            "status": "completed",
            "total_videos": session["total"],
            "message": "所有视频已传输完成"
        }
    
    # 获取当前视频
    video = session["videos"][current_index]
    video_path = os.path.join(VIDEO_DIR, video)
    
    # 更新索引
    session["current_index"] += 1
    
    # 检查是否最后一个
    is_last = current_index == session["total"] - 1
    if is_last:
        session["completed"] = True
    
    # 构建视频信息
    video_info = {
        "status": "streaming",
        "index": current_index,
        "filename": video,
        "size": os.path.getsize(video_path),
        "url": f"/videos/name/{video}",
        "total_videos": session["total"],
        "is_last": is_last
    }
    
    # 等待1秒模拟处理时间
    await asyncio.sleep(1)
    
    return video_info

@app.delete("/api/end-stream/{client_id}")
async def end_stream(client_id: str):
    """
    结束视频流会话，释放资源
    """
    if client_id in active_connections:
        del active_connections[client_id]
        return {"status": "success", "message": f"会话 {client_id} 已结束"}
    else:
        raise HTTPException(status_code=404, detail="无效的会话ID")

# 示例客户端代码
@app.get("/api/client-example")
async def client_example():
    """返回一个示例的Python客户端代码"""
    python_client = """
import requests
import time

# 服务器地址
SERVER_URL = "http://localhost:8000"

def start_video_stream():
    # 启动视频流
    response = requests.post(f"{SERVER_URL}/api/start-stream")
    if response.status_code != 200:
        print(f"错误: {response.json()['detail']}")
        return None
    
    data = response.json()
    client_id = data["client_id"]
    total_videos = data["total_videos"]
    
    print(f"开始视频流，总共 {total_videos} 个视频")
    return client_id, total_videos

def process_videos(client_id):
    # 处理视频流
    completed = False
    received_count = 0
    
    while not completed:
        # 获取下一个视频
        response = requests.get(f"{SERVER_URL}/api/next-video/{client_id}")
        if response.status_code != 200:
            print(f"错误: {response.json()['detail']}")
            break
        
        data = response.json()
        
        # 检查是否完成
        if data["status"] == "completed":
            print(f"所有视频处理完成，共 {data['total_videos']} 个")
            completed = True
            break
        
        # 处理视频
        received_count += 1
        video_url = f"{SERVER_URL}{data['url']}"
        print(f"接收到视频 {received_count}: {data['filename']}")
        print(f"下载链接: {video_url}")
        
        # 这里可以添加代码下载视频或进行其他处理
        # 例如：requests.get(video_url)
        
        # 如果是最后一个
        if data["is_last"]:
            print("这是最后一个视频")
    
    # 结束流
    end_stream(client_id)

def end_stream(client_id):
    # 主动结束视频流
    response = requests.delete(f"{SERVER_URL}/api/end-stream/{client_id}")
    if response.status_code == 200:
        print("视频流已结束")
    else:
        print(f"错误: {response.json()['detail']}")

if __name__ == "__main__":
    # 开始流程
    result = start_video_stream()
    if result:
        client_id, _ = result
        process_videos(client_id)
    """
    
    return StreamingResponse(
        content=BytesIO(python_client.encode('utf-8')),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=video_client_example.py"}
    )

if __name__ == "__main__":
    print(f"视频目录: {os.path.abspath(VIDEO_DIR)}")
    print(f"可用视频: {get_sorted_videos()}")
    uvicorn.run("main:app", port=8000, reload=True)
