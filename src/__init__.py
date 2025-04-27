from .LLM import Deepseek


def test_Deepseek(question="如何应对压力？"):
    llm = Deepseek()
    answer = llm.generate(question)
    print(answer)


class LLM:
    def init_model(self, model_name, model_path='', api_key=None, proxy_url=None,
                   prefix_prompt='''请用少于25个字回答以下问题\n\n'''):
        if model_name not in ['Deepseek', '直接回复 Direct Reply']:
            raise ValueError("model_name must be one of [ 'Deepseek', '直接回复 Direct Reply']")
        if model_name == 'Deepseek':
            llm = Deepseek()
        elif model_name == '直接回复 Direct Reply':
            llm = self
        llm.prefix_prompt = prefix_prompt
        return llm

    def chat(self, system_prompt, message, history):
        response = self.generate(message, system_prompt)
        history.append((message, response))
        return response, history

    def generate(self, question, system_prompt='system无效'):
        return question


if __name__ == '__main__':
    # 测试Qwen模型
    test_Deepseek()
