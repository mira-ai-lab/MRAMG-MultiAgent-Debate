import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)


class CLIPImageFilter:
    """使用 CLIP 模型筛选与查询最相关的 topk 图像"""
    
    def __init__(self, model_path: str = "/data2/qn/KGQA/models/openai/clip-vit-large-patch14"):
        """
        初始化 CLIP 模型
        
        Args:
            model_path: CLIP 模型路径
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """加载 CLIP 模型和处理器"""
        try:
            logger.info(f"正在加载 CLIP 模型: {self.model_path}")
            self.model = CLIPModel.from_pretrained(self.model_path).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_path)
            logger.info("CLIP 模型加载成功")
        except Exception as e:
            logger.error(f"CLIP 模型加载失败: {e}")
            raise
    
    def filter_images(self, query: str, all_img_paths: list, top_k: int = 5) -> list:
        """
        使用 CLIP 模型筛选与查询最相关的 topk 图像
        
        Args:
            query: 用户查询
            all_img_paths: 所有待筛选的图像路径列表
            top_k: 要返回的 topk 图像数量
            
        Returns:
            筛选后的 topk 图像路径列表（按相似度从高到低排序）
        """
        if self.model is None or self.processor is None:
            logger.warning("CLIP 模型未加载，返回原始图像列表")
            return all_img_paths[:top_k] if len(all_img_paths) > top_k else all_img_paths
        
        if not all_img_paths:
            return []
        
        if len(all_img_paths) <= top_k:
            return all_img_paths
        
        try:
            logger.info(f"开始使用 CLIP 筛选图像: 共 {len(all_img_paths)} 张，取 top {top_k} 张")
            
            # 加载所有图像
            images = []
            valid_img_paths = []
            
            for img_path in all_img_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    valid_img_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"加载图像失败 {img_path}: {e}")
                    continue
            
            if not valid_img_paths:
                return []
            
            # 准备输入 - 修复：分别处理文本和图像，移除不支持的padding参数
            text_inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
            image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            
            # 合并输入
            inputs = {**text_inputs, **image_inputs}
            
            # 获取相似度
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # 图像到文本的相似度
                probs = logits_per_image.squeeze().softmax(dim=0)  # 对图像维度进行softmax
            
            # 获取 topk 索引
            topk_indices = torch.topk(probs, min(top_k, len(valid_img_paths))).indices.tolist()
            
            # 获取 topk 图像路径
            filtered_img_paths = [valid_img_paths[i] for i in topk_indices]
            
            logger.info(f"CLIP 筛选完成: 从 {len(all_img_paths)} 张 → {len(filtered_img_paths)} 张")
            
            return filtered_img_paths
            
        except Exception as e:
            logger.error(f"CLIP 筛选图像失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return all_img_paths[:top_k] if len(all_img_paths) > top_k else all_img_paths


# 全局 CLIP 过滤器实例（懒加载）
_clip_filter = None


def get_clip_filter() -> CLIPImageFilter:
    """获取全局 CLIP 过滤器实例（单例模式）"""
    global _clip_filter
    if _clip_filter is None:
        _clip_filter = CLIPImageFilter()
    return _clip_filter