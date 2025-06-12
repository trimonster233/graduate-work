import os
import time
import gradio as gr
import glob

# 获取视频路径
video_dir = "/root/MuseTalk/results/v15/avatars/0"
video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

# 如果没有找到视频文件，显示错误
if not video_files:
    raise ValueError(f"在 {video_dir} 目录中没有找到MP4视频文件")

# 自动播放函数 - 每次返回一个视频，使用yield实现流式输出
def auto_play_videos():
    while True:
        for video_path in video_files:
            yield video_path, f"正在播放: {os.path.basename(video_path)}"
            time.sleep(3)

# 创建Gradio界面
with gr.Blocks() as app:
    gr.Markdown("# 视频自动播放器\n每3秒自动切换到下一个视频")
    
    # 状态显示
    status_text = gr.Textbox(label="当前状态", value="准备就绪")
    
    # 视频组件
    video_output = gr.Video(label="当前视频", interactive=False, autoplay=True)
    
    # 控制按钮
    auto_btn = gr.Button("自动播放")
    
    # 自动播放功能
    auto_btn.click(
        fn=auto_play_videos,
        inputs=[],
        outputs=[video_output, status_text],
        api_name="auto_play_videos"
    )

# 启动应用
if __name__ == "__main__":
    app.launch(share=True)  # share=True 允许外部访问