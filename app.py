import os
import time
import pdb
import re
import threading
import queue
import gradio as gr
import numpy as np
import sys
import subprocess
import openai
import json
from datetime import datetime


from musetalk.utils.blending import get_image_prepare_material, get_image_blending
import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
from argparse import Namespace
import shutil

from moviepy.editor import *
from transformers import WhisperModel

import json

def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print(child_path)


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range





@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id=0, bbox_shift=0, video_path="", batch_size=15, preparation=True, 
                extra_margin=10, parsing_mode="jaw", left_cheek_width=90, right_cheek_width=90):
        self.avatar_id = avatar_id
        self.bbox_shift = bbox_shift
        self.extra_margin = extra_margin
        self.parsing_mode = parsing_mode
        self.left_cheek_width = left_cheek_width
        self.right_cheek_width = right_cheek_width
        self.base_path = f"./results/v15/avatars/{avatar_id}"
        self.video_path = ""
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_answer_path = f"{self.avatar_path}/answer.wav"  # 大模型回复语音合成语音文件
        self.avatar_ask_path = f"{self.avatar_path}/ask.wav"  # 语音提问语音文件存储
        self.temp_video_path = None   # 用于存储临时视频路径
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "extra_margin": extra_margin,
            "parsing_mode": parsing_mode,
            "left_cheek_width": left_cheek_width,
            "right_cheek_width": right_cheek_width,
            "version": "v15"
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0


    def init(self, video_path):
        # 保存当前的参数设置
        current_bbox_shift = self.bbox_shift
        current_extra_margin = self.extra_margin
        current_parsing_mode = self.parsing_mode
        current_left_cheek_width = self.left_cheek_width
        current_right_cheek_width = self.right_cheek_width
        self.video_path = video_path
        self.frame_queue = queue.Queue()
        self.video_queue = queue.Queue()
        self.idx = 0
        if os.path.exists(self.avatar_path):
            shutil.rmtree(self.avatar_path)
        print("*********************************")
        print(f"  creating avator: {self.avatar_id}")
        print("*********************************")
        osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
        
        # 更新avatar_info字典
        self.avatar_info.update({
            "video_path": video_path,
            "bbox_shift": current_bbox_shift,
            "extra_margin": current_extra_margin,
            "parsing_mode": current_parsing_mode,
            "left_cheek_width": current_left_cheek_width,
            "right_cheek_width": current_right_cheek_width
        })
        
        # 还原参数设置
        self.bbox_shift = current_bbox_shift
        self.extra_margin = current_extra_margin
        self.parsing_mode = current_parsing_mode
        self.left_cheek_width = current_left_cheek_width
        self.right_cheek_width = current_right_cheek_width
        
        self.prepare_material()


    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox

            y2 = y2 + self.extra_margin
            y2 = min(y2, frame.shape[0])
            coord_list[idx] = [x1, y1, x2, y2]  # 更新coord_list中的bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        fp = FaceParsing(
            left_cheek_width=self.left_cheek_width,
            right_cheek_width=self.right_cheek_width
        )

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)
            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mode = self.parsing_mode
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def debug_inpainting(self, video_path, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width):
        """Debug inpainting parameters, only process the first frame"""
        # 更新Avatar的属性（如果提供了参数）
        
        # 使用当前Avatar实例的参数值
        self.bbox_shift = bbox_shift
        self.extra_margin = extra_margin    
        self.parsing_mode = parsing_mode
        self.left_cheek_width = left_cheek_width
        self.right_cheek_width = right_cheek_width
            
        # Set default parameters
        args_dict = {
            "result_dir": './results/debug',
            "fps": 25,
            "batch_size": 1,
            "output_vid_name": '',
            "use_saved_coord": True,
            "audio_padding_length_left": 2,
            "audio_padding_length_right": 2,
            "extra_margin": self.extra_margin,
            "parsing_mode": self.parsing_mode,
            "left_cheek_width": self.left_cheek_width,
            "right_cheek_width": self.right_cheek_width
        }
        debug_args = Namespace(**args_dict)

        # Create debug directory
        os.makedirs(debug_args.result_dir, exist_ok=True)

        # Read first frame
        if get_file_type(video_path) == "video":
            reader = imageio.get_reader(video_path)
            first_frame = reader.get_data(0)
            reader.close()
        else:
            first_frame = cv2.imread(video_path)
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        # Save first frame
        debug_frame_path = os.path.join(debug_args.result_dir, "debug_frame.png")
        cv2.imwrite(debug_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

        # Get face coordinates
        coord_list, frame_list = get_landmark_and_bbox([debug_frame_path], self.bbox_shift)
        bbox = coord_list[0]
        frame = frame_list[0]

        if bbox == coord_placeholder:
            return None, "No face detected, please adjust bbox_shift parameter"

        # Initialize face parser
        fp = FaceParsing(
            left_cheek_width=self.left_cheek_width,
            right_cheek_width=self.right_cheek_width
        )

        # Process first frame
        x1, y1, x2, y2 = bbox
        y2 = y2 + self.extra_margin
        y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # Generate random audio features
        random_audio = torch.randn(1, 50, 384, device=device, dtype=weight_dtype)
        audio_feature = pe(random_audio)

        # Get latents
        latents = vae.get_latents_for_unet(crop_frame)
        latents = latents.to(dtype=weight_dtype)

        # Generate prediction results
        pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_feature).sample
        recon = vae.decode_latents(pred_latents)

        # Inpaint back to original image
        res_frame = recon[0]
        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        combine_frame = get_image(frame, res_frame, [x1, y1, x2, y2], mode=self.parsing_mode, fp=fp)

        # Save results (no need to convert color space again since get_image already returns RGB format)
        debug_result_path = os.path.join(debug_args.result_dir, "debug_result.png")
        cv2.imwrite(debug_result_path, combine_frame)


        return cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR)


    
    def inference(self, audio_path):
        """执行推理过程，生成说话头像视频
        
        Args:
            audio_path: 输入的音频文件路径
            video_path: 输入的视频文件路径，若为None则使用当前Avatar的视频
            bbox_shift: 人脸边界框调整值，若为None则使用当前Avatar的设置
            extra_margin: 额外边距，若为None则使用当前Avatar的设置
            parsing_mode: 解析模式，若为None则使用当前Avatar的设置
            left_cheek_width: 左脸颊宽度，若为None则使用当前Avatar的设置
            right_cheek_width: 右脸颊宽度，若为None则使用当前Avatar的设置
            frame_callback: 用于处理流式输出帧的回调函数
            
        Returns:
            None，因为是流式处理所有帧
        """

            
        # 打印当前使用的参数
        print(f"使用参数: bbox_shift={self.bbox_shift}, extra_margin={self.extra_margin}, "
              f"parsing_mode={self.parsing_mode}, left_cheek_width={self.left_cheek_width}, "
              f"right_cheek_width={self.right_cheek_width}")
        
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)
        print("开始流式推理过程...")
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        # Extract audio features
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path,
                                                                               weight_dtype=weight_dtype)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,  # 使用固定值
            audio_padding_length_right=2,  # 使用固定值
        )
        print(f"音频处理完成: {audio_path}，耗时 {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        print(f"视频帧数: {video_num}")
        self.idx = 0


        gen = datagen(whisper_chunks,
                  self.input_latent_list_cycle,
                  self.batch_size)
        start_time = time.time()

        for i, (whisper_batch, latent_batch) in enumerate(
            tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

            pred_latents = unet.model(latent_batch,
                                  timesteps,
                                  encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred_latents)

            for res_frame in recon:
                # 实时处理每一帧
                bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
                ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
                x1, y1, x2, y2 = bbox
                try:
                    frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
                    mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
                    combine_frame = get_image_blending(ori_frame, frame_resized, bbox, mask, mask_crop_box)
                    self.frame_queue.put(combine_frame)
                except Exception as e:
                    print(f"处理帧时出错: {e}")
                self.idx += 1
        
        print(f'流式处理 {video_num} 帧完成，耗时: {time.time() - start_time}秒')
        print("流式推理过程完成\n")
        
        # 清理临时文件
        if os.path.exists(f"{self.avatar_path}/tmp"):
            shutil.rmtree(f"{self.avatar_path}/tmp")
            
        return None
    
    
    def chat_with_avatar(self, chatbot, text, video_path=None, bbox_shift=None, extra_margin=None, 
                        parsing_mode=None, left_cheek_width=None, right_cheek_width=None, api_key=None):
        """与虚拟形象聊天，生成对应的视频和语音响应，支持流式视频输出
        
        Args:
            chatbot: 聊天机器人对象
            text: 用户输入的文本
            video_path: 输入的视频文件路径，如果未上传则为None
            bbox_shift: 人脸边界框偏移
            extra_margin: 额外边距
            parsing_mode: 解析模式
            left_cheek_width: 左脸颊宽度
            right_cheek_width: 右脸颊宽度
            api_key: API密钥
        Returns:
            更新后的文本输入框内容，聊天历史，视频路径，错误信息
        """
                # 更新Avatar的属性（如果提供了参数）
        if bbox_shift is not None:
            self.bbox_shift = bbox_shift
        if extra_margin is not None:
            self.extra_margin = extra_margin
        if parsing_mode is not None:
            self.parsing_mode = parsing_mode
        if left_cheek_width is not None:
            self.left_cheek_width = left_cheek_width
        if right_cheek_width is not None:
            self.right_cheek_width = right_cheek_width
        error_msg = ""
        self.video_queue = queue.Queue()
        self.frame_queue = queue.Queue()

        

        # 生成回复文本
        response = process_text_with_openai(text, api_key=api_key)
        response_text = response

        
        
        # 添加到聊天历史
        updated_chatbot = chatbot + [[text, response_text]]
        # 更新全局聊天历史
        global chat_history
        chat_history.append([text, response_text])
        
        # 检查是否已上传视频
        if video_path is None or video_path == "":
            error_msg = "视频未上传,只进行文字对话"
            yield text, updated_chatbot, None, error_msg
            return
        else:
            # 先更新UI，显示AI回复
            yield "", updated_chatbot, None, "正在生成视频..."
            
            # 生成临时音频文件
            audio_path =  f"{avatar.avatar_path}/temp_audio.wav"


        if len(response_text) > 0:
            try:
                # 使用edge-tts生成音频
                import asyncio
                from src import TTSTalker
                tts = TTSTalker.TTSTalker()
                tts.predict(audio_path, response_text, voice_role)

                
                print(f"成功生成音频文件: {audio_path}")
            except ImportError as e:
                error_msg = f"TTSTalker导入出错, {str(e)}"
                yield text, updated_chatbot, None, error_msg
                return
            except Exception as e:
                error_msg = f"生成语音时出错: {str(e)}"
                yield text, updated_chatbot, None, error_msg
                return
            

            # 临时视频文件路径
            temp_output_path = f"{avatar.avatar_path}/temp_stream_video.mp4"
      
            
            # 确保目录存在
            os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)
            
            # 标记是否已创建视频写入器
            update_frequency = 3*fps  # 每收集多少帧更新一次UI,设计为3S的帧数
            
            
            # 计算预期的帧数（基于音频长度和FPS）
            try:
                import librosa
                audio_duration = librosa.get_duration(path=audio_path)
                expected_frames = int(audio_duration * fps)
                print(f"预期生成约 {expected_frames} 帧")
            except:
                expected_frames = 300  # 默认值
                print(f"无法计算音频长度，默认预期 {expected_frames} 帧")
            
            frame_count = 0
            # video_chunk合成线程函数
            def video_chunk_process():
                nonlocal frame_count, expected_frames
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 默认编码
                video_writer = None
                
                while frame_count < expected_frames:
                    frame = self.frame_queue.get()
                    frame_count += 1
                    if video_writer is None:
                        height, width = frame.shape[:2]
                        video_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
                    
                    # 写入每一帧到视频文件
                    video_writer.write(frame)
                    
                    # 如果收集到足够的帧，创建临时视频供UI更新
                    if frame_count % update_frequency == 0 or frame_count >= expected_frames:
                        # 关闭视频写入器，这样文件可以被读取
                        video_writer.release()
                        video_writer = None # 将写入器置空,因为每次release后,video_writer会失效
                        print(f"已处理 {frame_count} 帧")
                        
                        # 合并音频和视频为临时文件
                        update_video_path = f"{avatar.avatar_path}/update_{frame_count}.mp4"
                        try:
                            # 确保生成的视频文件正确
                            # 计算当前音频段对应的起始和结束时间点（秒）
                            if frame_count % update_frequency == 0:
                                start_time = ((frame_count - update_frequency) / fps) if frame_count > update_frequency else 0
                            else:
                                start_time = (frame_count - frame_count % update_frequency) / fps
                            end_time = frame_count / fps
                            print(f"开始生成前{frame_count}帧, 当前音频段对应的起始和结束时间点: {start_time} - {end_time}")
                            # 使用ffmpeg分割音频文件到对应的时间段
                            segment_audio_path = f"{avatar.avatar_path}/segment_audio_{frame_count}.wav"
                            audio_segment_cmd = f"ffmpeg -y -v warning -i {audio_path} -ss {start_time} -to {end_time} -c:a pcm_s16le {segment_audio_path}"
                            print(f"分割音频命令: {audio_segment_cmd}")
                            subprocess.run(audio_segment_cmd, shell=True, stderr=subprocess.PIPE, check=True)
                            
                            # 合并分段音频和视频
                            cmd_combine_audio = f"ffmpeg -y -v warning -i {segment_audio_path} -i {temp_output_path} -c:v copy -c:a aac -shortest {update_video_path}"
                            print(f"合并命令: {cmd_combine_audio}")
                            result = subprocess.run(cmd_combine_audio, shell=True, stderr=subprocess.PIPE, check=True)
                            print(f"命令执行结果: {result}")
                            
                            self.video_queue.put(update_video_path)
            
                        except Exception as e:
                            print(f"视频合成出错: {e}")
                


            
            chunk_thread = threading.Thread(target=video_chunk_process)
            inference_thread = threading.Thread(target=lambda: self.inference(audio_path))
            inference_thread.start()
            chunk_thread.start()
    
            ############ 播放线程 ############
            from math import ceil
            video_count = 0
            
            wait_time = update_frequency / fps
            expected_videos = ceil(expected_frames / update_frequency)
            
            while video_count < expected_videos:
                try:
                    # 设置超时时间，防止无限等待
                    video_path = self.video_queue.get()
                    video_count += 1
                    print(f"预期生成约 {expected_videos} 个视频, 当前生成第 {video_count} 个视频")
                    # 播放视频
                    yield text, updated_chatbot, video_path, f'播放视频{video_path}'
                    # 等待足够时间让前端播放视频
                    time.sleep(wait_time*1.1)

                except queue.Empty:
                    print("等待视频生成超时...")
                    # 继续循环，不退出
                except Exception as e:
                    print(f"播放视频时出错: {e}")
                if video_count >= expected_videos:
                    break
            

                    
            chunk_thread.join()
            inference_thread.join()
            print("推理线程结束")
                
            # 清理临时文件
            print("清理临时文件...")
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"删除音频文件: {audio_path}")
            
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
                print(f"删除临时视频: {temp_output_path}")
            
            # # 删除所有临时更新视频
            # for temp_file in glob.glob(f"{avatar.avatar_path}/update_*.mp4"):
            #     try:
            #         os.remove(temp_file)
            #         print(f"删除临时更新视频: {temp_file}")
            #     except Exception as e:
            #         print(f"删除文件失败: {temp_file}, 错误: {e}")
            
        else:
            error_msg = "生成的回复文本为空"
            yield text, updated_chatbot, None, error_msg


# 导入视频,检测该视频和原有视频是否相同,如果相同不管, 否则更新avatar
def check_video(video_path):
    print(f"当前视频路径: {video_path}")
    print(f"Avatar当前视频路径: {avatar.video_path}")
    
    if not video_path:
        print("警告: 视频路径为空")
        return video_path
        
    if not os.path.exists(video_path):
        print(f"警告: 视频文件不存在: {video_path}")
        return video_path
        
    if video_path != avatar.video_path:
        print("检测到新视频，开始初始化Avatar...")
        avatar.init(video_path)
        print("Avatar初始化完成")
    else:
        print("视频未变化，跳过初始化")
        
    return video_path


from rag.law_data_process import law_query


def get_prompt(query_str):
    global role
    prompt = ""
    if role == "无":
        prompt = "请用简洁明了的语言回答用户的问题"
    elif role == "程序员":
        prompt = "你是一个程序员,请用简洁明了的语言回答用户的问题"
    elif role == "律师":
        existing_answer = law_query(query_str)
        prompt =  '''
        你是一个律师现在你需要根据用户问题做出法律上的解答
        原始查询如下：{query_str}
        我们提供了现有答案：{existing_answer}
        我们有机会通过下面的更多上下文来完善现有答案（仅在需要时）。
        考虑到新的上下文，优化原始答案以更好地回答查询。 如果上下文没有用，请返回原始答案。
        Refined Answer:'''
    elif role == "医生":
        prompt = "你是一个医生,请用简洁明了的语言回答用户的问题"
    elif role == "厨师":
        prompt = "你是一个厨师,请用简洁明了的语言回答用户的问题"
    return prompt


def process_text_with_openai(text, api_key):
    """使用OpenAI处理文本输入，生成回复"""
    try:
        # 优先使用传入的api_key参数，否则尝试从环境变量获取
        key_to_use = api_key
        print(f"使用API密钥: {key_to_use}")
        if not key_to_use:
            return "请提供有效的API密钥"
        

        ##########构造prompt####################333
        # 创建消息列表，首先添加system消息
        messages = [{"role": "system", "content": "在30字以内尽可能简洁回答用户的问题"}]
        
        # 然后添加聊天历史记录(最多context_length条)
        for old_msg in chat_history[-context_length:]:
            messages.append({"role": "user", "content": old_msg[0]})
            if old_msg[1]:  # 确保有回复
                messages.append({"role": "assistant", "content": old_msg[1]})
        
        # 最后添加当前用户消息
        messages.append({"role": "user", "content": text})
            
        client = openai.OpenAI(
            api_key=key_to_use,
            base_url="https://api.deepseek.com/v1"
        )
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content
        # return "你好,我是雷军,我正在回答你的问题"
    except Exception as e:
        print(f"OpenAI API调用出错: {e}")
        return f"抱歉，我遇到了一些问题: {str(e)}"

def text_to_audio(text):
    """将文本转换为音频文件，返回文件路径"""
    # 为简化实现，直接创建一个空的音频文件
    # 在实际应用中，这里应使用TTS服务生成语音
    temp_audio_path = f"./results/answer.wav"
    # 确保目录存在
    os.makedirs(os.path.dirname(temp_audio_path), exist_ok=True)
    
    # 这里应该有TTS代码，暂时跳过
    # 在实际项目中应使用适当的TTS引擎将text转换为语音
    
    return temp_audio_path

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default="/usr/bin/ffmpeg", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="/root/autodl-tmp/models/musetalk/musetalk.json",
                        help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="/root/autodl-tmp/models/musetalk/pytorch_model.bin",
                        help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="/root/autodl-tmp/models/whisper",
                        help="Directory containing Whisper model")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=15, help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")     
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--context_length", type=int, default=10, help="聊天上下文最大长度")
    args = parser.parse_args()

        # 全局聊天历史记录
    chat_history = []

    # 定义音频角色字典，包含ID和名称的对应关系
    voice_roles = {
        "智萱":101023,
        "智皓": 101024,
        "智薇": 101025,
        "智蓓": 101033,
        "四川女声": 101040,
        "英文男声": 101050,
        "英文女声": 101051,
        "东北男声": 101056,
        "粤语女声": 101019,
        "日语女声": 101057,
        "男童声": 101015,
        "女童声": 101016
    }

    roles = ['无', '程序员', '医生', '律师', '厨师']
    # 创建下拉框选项列表
    voice_role_choices = list(voice_roles.keys())
    # 定义默认语音角色
    voice_role = list(voice_roles.values())[0]  # 默认使用第一个语音角色的ID
    role = roles[0]
    total_start_time = time.time()
    fps = args.fps
    context_length = args.context_length
    print(f"当前帧率: {fps}")
    #########加载模型#############################################################################
    print("=" * 50)
    print("开始加载模型")
    print("=" * 50)

    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        # Choose path separator based on operating system
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # Load model weights
    print("\n1. 开始加载模型权重")
    start_time = time.time()
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    print(f"模型权重加载完成，耗时: {time.time() - start_time:.2f}秒")

    timesteps = torch.tensor([0], device=device)

    print("\n2. 开始转换模型精度")
    start_time = time.time()
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    print(f"模型精度转换完成，耗时: {time.time() - start_time:.2f}秒")

    # Initialize audio processor and Whisper model
    print("\n3. 开始加载音频处理器和Whisper模型")
    start_time = time.time()
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    print(f"音频模型加载完成，耗时: {time.time() - start_time:.2f}秒")

    # Initialize face parser with configurable parameters based on version
    print("\n4. 开始初始化人脸解析器")
    start_time = time.time()

    print(f"人脸解析器初始化完成，耗时: {time.time() - start_time:.2f}秒")


    print("=" * 50)
    print("模型加载完成")
    print("=" * 50)
    ########加载模型结束#############################################################################



    # 创建Avatar实例，使用默认参数
    avatar = Avatar(
        avatar_id=0,
        video_path="",
        batch_size=args.batch_size,
        preparation=True,
        parsing_mode=args.parsing_mode,
    )
    



    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Sidebar():
                # 设置区域
                with gr.Tab("基本设置"):
                    video = gr.Video(label="数字人视频", sources=['upload'])
                    voice_choice = gr.Dropdown(label="语音", choices=voice_role_choices, value=voice_role_choices[0])
                    role_choice = gr.Dropdown(label="对话角色", choices=roles, value=roles[0])
                    api_key = gr.Textbox(label="API Key", value="sk-23f4ee59bc8244c6bf8c3dc1d742304f", type="password", placeholder="在此输入您的OpenAI API Key")
                    
                    with gr.Row():
                        test_btn = gr.Button("测试数字人")
                
                # 高级参数区域
                with gr.Tab("高级参数"):
                    bbox_shift = gr.Number(label="人脸中心偏移量", value=0)
                    extra_margin = gr.Slider(label="额外边缘宽度", minimum=0, maximum=40, value=10, step=1)
                    parsing_mode = gr.Radio(label="融合模式", choices=["jaw", "raw"], value="jaw")
                    left_cheek_width = gr.Slider(label="左脸宽度", minimum=20, maximum=160, value=90, step=5)
                    right_cheek_width = gr.Slider(label="右脸宽度", minimum=20, maximum=160, value=90, step=5)
                    bbox_shift_scale = gr.Textbox(label="参数说明", lines = 6, value="'左脸宽度'和'右脸宽度'参数决定了融合模式为'jaw'时左右脸部编辑的范围。'额外边缘宽度'参数决定了下巴的移动范围。用户可以自由调整这三个参数以获得更好的嵌入效果。")
            
            # 主显示区域
            with gr.Column():
                with gr.Tab("数字人对话"):
                    with gr.Row():
                        # 左侧视频区域
                        with gr.Column(scale=1):
                            avatar_video = gr.Video(
                                label="数字人", 
                                show_download_button = False, 
                                interactive=False, 
                                elem_id="avatar_video",
                                width=400, 
                                height=600, 
                                autoplay=True)
                        
                        # 右侧聊天区域
                        with gr.Column(scale=2):
                            # 聊天历史
                            chatbot = gr.Chatbot(label="对话历史", height=380)
                            
                            # 错误信息显示
                            error_output = gr.Textbox(label="状态信息", interactive=False)
                            
                            # 输入区域
                            with gr.Row():
                                text_input = gr.Textbox(label="在此输入文字", placeholder="请输入您想对数字人说的话...", lines=2)
                                audio_input = gr.Audio(label="语音输入", sources=["microphone"], type="filepath")
                            
                            with gr.Row():
                                submit_btn = gr.Button("发送", variant="primary")
                                clear_btn = gr.Button("清空对话")
                
                with gr.Tab("测试区域"):
                    debug_image = gr.Image(label="测试图像效果")

        # 事件响应
        video.change(
            fn=check_video, inputs=[video], outputs=[video]
        )
        
        def update_voice_role(voice_name):
            global voice_role
            voice_role = voice_roles.get(voice_name)  
            print(f"语音角色已更新为: {voice_name}, ID: {voice_role}")
            return voice_name
        

        def update_role(role_name):
            global role
            role = role_name
            print(f"对话角色已更新为: {role_name}")
            return role_name
        
        voice_choice.select(
            fn=update_voice_role,
            inputs=[voice_choice],
            outputs=[voice_choice]
        )

        role_choice.select(
            fn=update_role,
            inputs=[role_choice],
            outputs=[role_choice]
        )

        
        # 测试按钮事件
        test_btn.click(
            fn=avatar.debug_inpainting,
            inputs=[video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[debug_image]
        )
        
        # 提交按钮事件 - 注意我们使用了stream参数来实现流式输出
        submit_btn.click(

            fn=avatar.chat_with_avatar,
            inputs=[chatbot, text_input, video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width, api_key],
            outputs=[text_input, chatbot, avatar_video, error_output],
            api_name="chat_with_avatar",
            scroll_to_output=True,
            show_progress=True
        )
 
        # 清空按钮事件
        clear_btn.click(
            fn=lambda: ("", [], None, ""),
            inputs=[],
            outputs=[text_input, chatbot, avatar_video, error_output]
        )

        # 语音输入处理
        def process_audio_input(audio_path, chatbot, video_ref, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width, api_key):
            import whisper
            model = whisper.load_model("base")
            result = whisper.transcribe(model, audio_path, language="zh")
            recognized_text = result["text"]
            if not audio_path:
                return "", chatbot, None, "未检测到语音输入"
                
            
            # 返回识别的文本到文本输入框，其他值保持不变
            # 然后用户可以点击发送按钮来继续对话
            return recognized_text, chatbot, None, "语音已转换为文本，请点击发送按钮继续"
        
        audio_input.change(
            fn=process_audio_input,
            inputs=[audio_input, chatbot, video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width, api_key],
            outputs=[text_input, chatbot, avatar_video, error_output]
        )

    # Check ffmpeg and add to PATH
    if not fast_check_ffmpeg():
        print(f"Adding ffmpeg to PATH: {args.ffmpeg_path}")
        # According to operating system, choose path separator
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")


    # Start Gradio application
    demo.queue().launch(
        share=args.share,
        debug=True,
        server_name=args.ip,
        server_port=args.port
    )

