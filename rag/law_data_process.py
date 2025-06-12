import pandas as pd
import chromadb
import requests
import uuid
import os

def embedding(text):
    res = requests.post(
        url="http://127.0.0.1:11434/api/embeddings",
        json={
            "model":"nomic-embed-text",
            "prompt": text
        }
    )

    embedding = res.json()['embedding']

    return embedding

def law_add(batch_size=100, total_rows=1000):
    # 确保数据库目录存在
    os.makedirs('db', exist_ok=True)
    
    # 创建或获取Chroma集合
    client = chromadb.PersistentClient(path="db/chroma_demo")
    collection = client.get_or_create_collection(name = "law_collection")
    
    # 读取CSV数据
    df = pd.read_csv("data/law.csv")
    if total_rows == -1:
        total_rows = len(df)
        
    # 分批处理数据
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        print(f"处理数据批次: {start_idx+1} 到 {end_idx} (共 {total_rows} 条)")
        
        batch_df = df.iloc[start_idx:end_idx]
        
        # 准备数据
        ids = []
        full_contents = []
        
        for _, row in batch_df.iterrows():
            # 生成唯一ID
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            
            # 提取字段内容
            title = row['title'] if not pd.isna(row['title']) else ""
            question = row['question'] if not pd.isna(row['question']) else ""
            reply = row['reply'] if not pd.isna(row['reply']) else ""
            
            # 组合完整文档内容
            full_content = f"标题: {title}\n问题: {question}\n回答: {reply}"
            full_contents.append(full_content)
        
        # 生成内容的嵌入向量
        print(f"为 {len(full_contents)} 个内容生成嵌入向量...")
        embeddings = [embedding(full_content) for full_content in full_contents]
        
        # 将完整内容和ID存入Chroma
        print("添加到Chroma数据库...")
        collection.add(
            ids=ids,
            documents=full_contents,
            embeddings=embeddings
        )
        
        print(f"批次处理完成，已添加 {len(ids)} 条记录")
    
    print(f"总共处理完成 {total_rows} 条法律数据")

def law_query(query, n_results=2):
    # 连接到Chroma
    client = chromadb.PersistentClient(path="db/chroma_demo")
    collection = client.get_or_create_collection(name = "law_collection")
    
    # 生成查询的嵌入向量
    query_embedding = embedding(query)
    
    # 在Chroma中搜索最相似的内容
    results = collection.query(
        query_embeddings=[query_embedding],
        query_texts=[query], 
        n_results=n_results
    )
    
    # 返回匹配结果
    return {
        "matched_ids": results['ids'][0],
        "contents": results['documents'][0]
    }


### ollama serve
if __name__ == "__main__":
    # 运行数据加载
    # law_add(batch_size=100, total_rows=1000)
    
    # 测试查询
    result = law_query("盗窃罪怎么判", n_results=10)
    print("\n查询结果:")
    for content in result["contents"]:
        print("-" * 30)
        print(content)