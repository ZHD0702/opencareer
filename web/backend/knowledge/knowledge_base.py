"""Knowledge Base - 知识库系统

使用 FAISS 向量数据库实现文档检索和问答功能。
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
import numpy as np

logger = logging.getLogger(__name__)

# 默认嵌入模型配置
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_FAISS_INDEX_PATH = "knowledge/faiss_index"
DEFAULT_DOCUMENTS_PATH = "knowledge/documents"


class KnowledgeBase:
    """知识库管理类"""
    
    _instance: Optional['KnowledgeBase'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._index = None
            self._documents = []
            self._embedding_model = None
            self._index_path = Path(DEFAULT_FAISS_INDEX_PATH)
            self._documents_path = Path(DEFAULT_DOCUMENTS_PATH)
            self._load_index()
    
    def _load_embedding_model(self):
        """加载嵌入模型"""
        try:
            from openai import OpenAI
            
            self._embedding_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("OpenAI embedding client initialized")
            return True
        except ImportError:
            logger.warning("OpenAI library not available")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize embedding client: {e}")
            return False
    
    def _load_index(self):
        """加载或创建 FAISS 索引"""
        if self._index_path.exists() and self._index_path.is_dir():
            try:
                self._index = faiss.read_index(str(self._index_path / "index.faiss"))
                self._documents = self._load_documents()
                logger.info(f"Loaded FAISS index with {len(self._documents)} documents")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """创建新的 FAISS 索引"""
        self._index = faiss.IndexFlatL2(1536)  # text-embedding-3-small 输出维度
        self._documents = []
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._documents_path.mkdir(parents=True, exist_ok=True)
        logger.info("Created new FAISS index")
    
    def _load_documents(self) -> List[Dict[str, Any]]:
        """加载文档元数据"""
        metadata_file = self._index_path / "documents.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    
    def _save_index(self):
        """保存索引和文档元数据"""
        try:
            faiss.write_index(self._index, str(self._index_path / "index.faiss"))
            with open(self._index_path / "documents.json", "w", encoding="utf-8") as f:
                json.dump(self._documents, f, ensure_ascii=False, indent=2)
            logger.info("FAISS index saved")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _embed_text(self, text: str) -> np.ndarray:
        """生成文本嵌入向量"""
        if not self._embedding_client:
            if not self._load_embedding_model():
                raise RuntimeError("Embedding model not available")
        
        response = self._embedding_client.embeddings.create(
            input=text,
            model=DEFAULT_EMBEDDING_MODEL
        )
        return np.array([response.data[0].embedding], dtype=np.float32)
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None):
        """
        添加文档到知识库
        
        Args:
            content: 文档内容
            metadata: 文档元数据（如标题、来源、类别等）
        """
        if not content.strip():
            return
        
        try:
            embedding = self._embed_text(content)
            self._index.add(embedding)
            
            document = {
                "content": content,
                "metadata": metadata or {},
                "id": len(self._documents)
            }
            self._documents.append(document)
            
            # 每添加 10 个文档保存一次
            if len(self._documents) % 10 == 0:
                self._save_index()
            
            logger.info(f"Added document #{len(self._documents)}")
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
    
    def add_documents_from_directory(self, directory: str):
        """从目录批量添加文档"""
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return
        
        for file_path in dir_path.rglob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                metadata = {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "type": "txt"
                }
                self.add_document(content, metadata)
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        self._save_index()
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索知识库
        
        Args:
            query: 搜索查询
            k: 返回结果数量
        
        Returns:
            匹配的文档列表，按相似度排序
        """
        if self._index.ntotal == 0:
            return []
        
        try:
            query_embedding = self._embed_text(query)
            distances, indices = self._index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self._documents):
                    doc = self._documents[idx]
                    results.append({
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "similarity": float(1 - distances[0][i] / 2)  # 归一化相似度
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def query(self, query: str, max_results: int = 3) -> str:
        """
        查询知识库并返回格式化结果
        
        Args:
            query: 查询问题
            max_results: 最大返回结果数
        
        Returns:
            格式化的回答文本
        """
        results = self.search(query, k=max_results)
        
        if not results:
            return "知识库中未找到相关信息。"
        
        response_parts = ["根据知识库，以下是相关信息："]
        
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("filename", "未知来源")
            response_parts.append(f"\n{i}. 【{source}】")
            response_parts.append(f"{result['content'][:500]}...")
        
        return "\n".join(response_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return {
            "document_count": len(self._documents),
            "index_size": self._index.ntotal if self._index else 0,
            "is_ready": self._embedding_client is not None
        }
    
    def clear(self):
        """清空知识库"""
        self._create_new_index()
        self._save_index()
        logger.info("Knowledge base cleared")


# 全局知识库实例
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """获取全局知识库实例"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base