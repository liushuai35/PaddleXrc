# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json, glob, os

num = 30
dir_name = "all_gt"

# 示例数据：多个JSON文件的路径
json_files = glob.glob(
    f"/home/shuai.liu01/PaddleXrc/api_examples/pipelines/{dir_name}/*.json"
)
output_path = f"output.json"

# 初始化结果列表
result = []

json_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
for file_path in json_files:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

        # 提取 parsing_res_list
        parsing_res_list = data["parsing_res_list"]

        # 提取 block_size 和 page_idx
        block_size = data["block_size"]
        page_idx = data["page_index"]

        sub_labels = []
        sub_bboxes = []
        sub_contents = []
        sub_indexes = []

        # 处理每个解析结果
        for item in parsing_res_list:
            if item.get("index"):
                sub_labels.append(item["block_label"])
                sub_bboxes.append(item["block_bbox"])
                sub_contents.append(item["block_content"])
                sub_indexes.append(item["index"])

        result.append(
            {
                "block_size": block_size,
                "page_idx": page_idx,
                "sub_labels": sub_labels,
                "sub_bboxes": sub_bboxes,
                "sub_contents": sub_contents,
                "sub_indices": sub_indexes,
            }
        )

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False, indent=4)
