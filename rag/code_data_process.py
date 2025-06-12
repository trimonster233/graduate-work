import pandas as pd
import chromadb
import requests
import uuid
import sqlite3
import os
import json

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

def create_sqlite_db():
    """创建SQLite数据库存储完整内容"""
    conn = sqlite3.connect('db/code_content.db')
    cursor = conn.cursor()
    
    # 创建表存储法律数据内容
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS code_contents (
        id TEXT PRIMARY KEY,
        title TEXT,
        question TEXT,
        reply TEXT,
        full_content TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print("SQLite数据库初始化完成")

def code_add(batch_size=100, total_rows=1000):
    # 确保数据库目录存在
    os.makedirs('db', exist_ok=True)
    
    # 创建SQLite数据库
    create_sqlite_db()
    conn = sqlite3.connect('db/code_content.db')
    cursor = conn.cursor()
    
    # 创建或获取Chroma集合
    client = chromadb.PersistentClient(path="db/chroma_demo")
    collection = client.get_or_create_collection(name = "code_title_collection")
    
    # 读取CSV数据
    df = pd.read_csv("data/code.csv")
    if total_rows == -1:
        total_rows = len(df)
    # 分批处理数据
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        print(f"处理数据批次: {start_idx+1} 到 {end_idx} (共 {total_rows} 条)")
        
        batch_df = df.iloc[start_idx:end_idx]
        
        # 准备数据
        ids = []
        titles = []
        documents = []
        
        for _, row in batch_df.iterrows():
            # 生成唯一ID
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            
            # 提取字段内容
            title = row['title'] if not pd.isna(row['title']) else ""
            question = row['question'] if not pd.isna(row['question']) else ""
            reply = row['reply'] if not pd.isna(row['reply']) else ""
            
            # 标题作为向量化对象
            titles.append(title)
            
            # 组合完整文档内容
            full_content = f"标题: {title}\n问题: {question}\n回答: {reply}"
            documents.append(full_content)
            
            # 存储到SQLite
            cursor.execute(
                "INSERT INTO code_contents (id, title, question, reply, full_content) VALUES (?, ?, ?, ?, ?)",
                (doc_id, title, question, reply, full_content)
            )
        
        # 生成标题的嵌入向量
        print(f"为 {len(titles)} 个标题生成嵌入向量...")
        embeddings = [embedding(title) for title in titles]
        
        # 将标题和ID存入Chroma
        print("添加到Chroma数据库...")
        collection.add(
            ids=ids,
            documents=titles,  # 只存储标题
            embeddings=embeddings
        )
        
        # 提交SQLite事务
        conn.commit()
        print(f"批次处理完成，已添加 {len(ids)} 条记录")
    
    # 关闭数据库连接
    conn.close()
    print(f"总共处理完成 {total_rows} 条法律数据")

def code_query(query, n_results=2):
    # 连接到Chroma
    client = chromadb.PersistentClient(path="db/chroma_demo")
    collection = client.get_or_create_collection(name = "code_title_collection")
    
    # 生成查询的嵌入向量
    query_embedding = embedding(query)
    
    # 在Chroma中搜索最相似的标题
    chroma_results = collection.query(
        query_embeddings=[query_embedding],
        query_texts=[query], 
        n_results=n_results
    )
    
    # 获取匹配的ID
    matched_ids = chroma_results['ids'][0]
    
    # 从SQLite中获取完整内容
    conn = sqlite3.connect('db/code_content.db')
    cursor = conn.cursor()
    
    results = []
    for doc_id in matched_ids:
        cursor.execute("SELECT full_content FROM code_contents WHERE id = ?", (doc_id,))
        result = cursor.fetchone()
        if result:
            results.append(result[0])
    
    conn.close()
    
    return {
        "matched_ids": matched_ids,
        "contents": results
    }

if __name__ == "__main__":
    # 运行数据加载
    # code_add(batch_size=100, total_rows=-1)
    
    # 测试查询
    result = code_query("醉驾怎么判")
    print("\n查询结果:")
    for content in result["contents"]:
        print("-" * 30)
        print(content)