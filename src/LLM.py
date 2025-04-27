import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import deepseek_api

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from openai import OpenAI  # 注意新版导入方式

class Deepseek:
    def __init__(self, prefix_prompt='''请用少于25个字回答以下问题\n\n''', url="https://api.deepseek.com/v1"):
        self.client = OpenAI(api_key=deepseek_api, base_url=url)
        self.model = "deepseek-chat"

        # 全局设定的prompt
        self.prefix_prompt = prefix_prompt
        self.history = []

    def predict_api(self, question):
        messages = []
        messages.append({"role": "system", "content": self.prefix_prompt})
        for user_msg, bot_msg in self.history:
            messages.extend([
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": bot_msg}
            ])

        messages.append({"role": "user", "content": question})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return response.choices[0].message.content

    def generate(self, question, system_prompt=""):
        return self.predict_api(question)

    def message_to_prompt(self, message, system_prompt=""):
        system_prompt = self.prefix_prompt + system_prompt
        for interaction in self.history:
            user_prompt, bot_prompt = str(interaction[0]).strip(' '), str(interaction[1]).strip(' ')
            system_prompt = f"{system_prompt} User: {user_prompt} Bot: {bot_prompt}"
        prompt = f"{system_prompt} ### Instruction:{message.strip()}  ### Response:"
        return prompt

    def chat(self, system_prompt, message, history):
        self.history = history
        prompt = self.message_to_prompt(message, system_prompt)
        response = self.generate(prompt)
        self.history.append([message, response])
        return response, self.history

    def clear_history(self):
        # 清空历史记录
        self.history = []


if __name__ == '__main__':
    llm = Deepseek()
    try:
        answer = llm.generate("如何应对压力？")
    except Exception as e:
        print("ASR Error: ", e)
    print(answer)
