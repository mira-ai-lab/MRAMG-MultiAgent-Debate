import json
def get_image_info(doc_name, img_list):
    """
    返回:
        caption_dict: {img_name: caption}
        img_paths:    [path1, path2, ...]
    """

    json_path = f"MRAMG-Bench/IMAGE/IMAGE/images_info/{doc_name}_imgs_collection.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    # 1️⃣ caption dict
    caption_dict = {
        img: data[img]["image_caption"]
        for img in img_list
    }

    # 2️⃣ img paths（doc_name 改成大写）
    doc_upper = doc_name.upper()

    img_paths = [
        f"MRAMG-Bench/IMAGE/IMAGE/images/{doc_upper}/{data[img]['image_path']}"
        for img in img_list
    ]

    return caption_dict, img_paths

def build_prompt_from_chroma(doc_name, chunks):
    # import pdb; pdb.set_trace()
    docs = chunks["documents"][0]
    metas = chunks["metadatas"][0]
    
    context_parts = []
    image_captions = []
    all_img_paths = []
    
    global_img_id = 1
    img_name_to_id = {}
    
    for i, (text, meta) in enumerate(zip(docs, metas)):
        
        img_str = meta.get("include_img_list", "")
        
        # 如果没有图片
        if not img_str.strip():
            context_parts.append(f"[context_{i+1}] {text}")
            continue
        
        img_list = img_str.split(",")
        caption_dict, img_paths = get_image_info(doc_name, img_list)
        
        # 逐个替换 <PIC>
        for img_name, img_path in zip(img_list, img_paths):
            if img_name not in img_name_to_id:
                img_name_to_id[img_name] = global_img_id
                image_captions.append(
                    f"[img{global_img_id}_caption] {caption_dict[img_name]}"
                )
                all_img_paths.append(img_path)
                global_img_id += 1

            img_id = img_name_to_id[img_name]
            
            # 替换一个 <PIC>
            text = text.replace("<PIC>", f"<img{img_id}>", 1)
        
        context_parts.append(f"[context_{i+1}] {text}")
        
    # 加一层判断，image_captions和all_img_paths长度是否一致
    if len(image_captions) != len(all_img_paths):
        raise ValueError("image_captions和all_img_paths长度不一致")
    
    context_block = "\n".join(context_parts)
    # caption_block = "\n".join(image_captions)
    
    return context_block, image_captions, all_img_paths, img_name_to_id
