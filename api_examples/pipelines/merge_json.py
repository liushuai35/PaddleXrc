import json,glob,os
num = 30
dir_name = "all_gt"

# 示例数据：多个JSON文件的路径
json_files = glob.glob(
        f"/home/user/liushuai/PaddleXrc/api_examples/pipelines/{dir_name}/{num}/*.json"
)
output_path = f'/home/user/liushuai/DocBench-100/{num}/input_{num}.json'

# 初始化结果列表
result = []

json_files.sort(key=lambda x:int(os.path.basename(x).split('_')[1]))
for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        # 提取 parsing_res_list
        parsing_res_list = data['parsing_res_list']
        
        # 提取 block_size 和 page_idx
        block_size = data['block_size']
        page_idx = data['page_index']
        
        sub_labels = []
        sub_bboxes = []
        sub_contents = []
        
        # 处理每个解析结果
        for item in parsing_res_list:
            sub_labels.append(item["block_label"])
            sub_bboxes.append(item["block_bbox"])
            sub_contents.append(item["block_content"])
            
        result.append({
            "block_size": block_size,
            "page_idx": page_idx,
            "sub_labels": sub_labels,
            "sub_bboxes": sub_bboxes,
            "sub_contents": sub_contents,
        })

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False, indent=4)