# 外卖客服智能回复系统
本项目是一个面向外卖商家的智能客服 Chatbot 系统，结合真实用户评论数据，支持：
- 评论情绪识别
- 方面关键词抽取
- FAQ 智能匹配
- 回复自动生成
- 情绪驱动的转人工逻辑

 项目目标：减轻客服负担，提升响应效率，支持多商家部署复用，适用于中小商家或平台统一客服系统。



## 项目系统流程
<img src="https://github.com/user-attachments/assets/383a7e64-51f0-4ace-b69b-e3b1ae337bbb" width="600">

流程说明：
- 餐前评论优先尝试 FAQ 向量匹配，匹配失败则进入人工处理；
- 餐后评论通过方面识别与情绪识别模块，判断是否调用大模型回复或转人工；
- 所有对话记录都会进入 Memory 模块，支持多轮上下文记忆。

## 使用演示

以下为项目运行效果截图（Gradio 前端界面）：  


<img src="https://github.com/user-attachments/assets/146091a3-6c65-4f39-9012-a2a70095eb10" width="1000">  


用户评论：今天的菜很难吃！  
系统输出：真的很抱歉今天的菜没有让您满意，您的反馈我们一定会认真记录并努力改进。  

## 环境依赖
```bash
pip install transformers langchain chromadb gradio
```
- 推荐使用 Python 3.9 及以上版本

- 推荐使用具备 GPU 的本地环境（本地部署 Qwen 或其他大模型）

- 也支持通过 API 调用大语言模型（如 GPT / 硅基流动 Qwen / MiniMax 等）

## 快速启动
```bash
python ui.py
```
## FAQ 数据格式示例
```json
[
  {
    "question": "营业时间是多久？",
    "answer": "您好, 我们店的营业时间为11:00 - 22:00～"
  },
]
```

## 当前缺点
响应速度偏慢（20s~60s）
原因可能包括：
- 使用云端 API 推理存在网络延迟
- 每轮请求均加载上下文，未做缓存或上下文压缩优化
  
后续可优化：
- 在多轮对话中引入缓存机制
- 使用更轻量化本地模型（如 ChatGLM2 / Qwen1.5-int4）

## License
本项目仅供学习研究使用，禁止商用。如需合作请联系作者。

