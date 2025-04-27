import os
import tempfile

import json
import types
import base64
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tts.v20190823 import tts_client, models




class TTSTalker:
    def __init__(self) -> None:
        # 从环境变量获取凭证，如果没有则使用默认值
        secret_id = os.environ.get("TTS_SECRET_ID", "")
        secret_key = os.environ.get("TTS_SECRET_KEY", "")
        
        if not secret_id or not secret_key:
            print("警告: 未设置TTS_SECRET_ID或TTS_SECRET_KEY环境变量")
            
        cred = credential.Credential(secret_id=secret_id, secret_key=secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tts.tencentcloudapi.com"

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        # 实例化要请求产品的client对象,clientProfile是可选的
        self.client = tts_client.TtsClient(cred, "", clientProfile)

    def predict(self, file_path, TEXT, VOICE = 0, SPEED = 0, VOLUME = 0):
        req = models.TextToVoiceRequest()
        params = {
            "Text": TEXT,
            "SessionId": "1",
            "VoiceType": VOICE,
            "Speed": SPEED,
            "Volume": VOLUME
        }
        req.from_json_string(json.dumps(params))
        resp = self.client.TextToVoice(req)
        resp_json = json.loads(resp.to_json_string())

        # 提取音频数据并解码
        audio_data = resp_json.get("Audio")  # 获取 Base64 编码的音频数据
        if not audio_data:
            raise ValueError("未获取到音频数据")

        audio_bytes = base64.b64decode(audio_data)
        with open(file_path, "wb") as f:
            f.write(audio_bytes)

        print(f"文字转音频文件导入：{file_path}")

        return file_path

if __name__ == "__main__":
    tts = TTSTalker()
    tts.predict("test.wav", "你好")

