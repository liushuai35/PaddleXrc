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

# from paddlex import create_pipeline

# pipeline = create_pipeline(pipeline="layout_parsing_v2",device='gpu:0')

# output = pipeline.predict(
#     "/home/user/liushuai/DocBench-100/70/simple.pdf",
#     # "/mnt/shuailiu35/eval_layout_order/70/input.pdf",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_common_ocr=True,
#     use_seal_recognition=True,
#     use_table_recognition=True,
# )

# for res in output:
#     # res.print()
#     res.save_to_img("./all_gt/70")
#     res.save_to_json("./all_gt/70")
    # res.save_to_xlsx("./output")
    # res.save_to_html("./output")
    # res.save_to_markdown("./output1")


# import os
# from glob import glob
# from paddlex import create_pipeline

# # 创建pipeline实例
# pipeline = create_pipeline(pipeline="layout_parsing_v2", device='gpu:0')

# # 输入和输出目录定义
# input_dir = "/home/shuai.liu01/PaddleXrc/inputs"  # 替换为实际输入目录路径
# output_dir = "./output"  # 替换为实际输出目录路径

# # 获取所有pdf文件路径
# # pdf_files = glob(os.path.join(input_dir, "*.pdf"))
# pdf_files = ["/home/shuai.liu01/PaddleXrc/inputs/single_column.pdf"]

# import json
# with open("/home/shuai.liu01/PaddleXrc/input_jsons/input_single_column.json", "r", encoding="utf-8") as file:
#     data = json.load(file)

# # 初始化一个列表来保存错误信息
# errors = []

# for pdf_file in pdf_files:
#     # try:
#         # 获取文件基础名（不带扩展名）
#         base_name = os.path.splitext(os.path.basename(pdf_file))[0]

#         # 为当前pdf文件创建输出子目录
#         output_subdir = os.path.join(output_dir, base_name)
#         os.makedirs(output_subdir, exist_ok=True)

#         # 执行预测
#         output = pipeline.predict(
#             pdf_file,
#             use_doc_orientation_classify=False,
#             use_doc_unwarping=False,
#             use_common_ocr=True,
#             use_seal_recognition=True,
#             use_table_recognition=True,
#         )

#         for res in output:
#             # 保存结果到对应的输出子目录
#             res.save_to_img(output_subdir)
#             res.save_to_json(output_subdir)
#             # 如果需要保存其他格式的结果，取消下面注释行的注释并根据需要调整路径
#             # res.save_to_xlsx(output_subdir)
#             # res.save_to_html(output_subdir)
#             # res.save_to_markdown(output_subdir)

#     # except Exception as e:
#     #     print(e)
#     #     # 记录错误信息
#     #     errors.append((pdf_file, str(e)))
#     #     print(f"处理文件 {pdf_file} 时发生错误，已跳过: {e}")

# # 所有文件处理完毕后，打印出错的文件和错误信息
# if errors:
#     print("\n以下文件在处理过程中发生了错误:")
#     for file_path, error_msg in errors:
#         print(f"文件: {file_path}, 错误: {error_msg}")
# else:
#     print("所有文件处理成功，没有错误发生。")


from paddlex.inference.pipelines.layout_parsing.utils import direct_test

# keys = ["1andmore_column", "single_column", "double_column", "three_column"]
# for key in keys:
#     direct_test(
#         f"/home/shuai.liu01/PaddleXrc/input_jsons/input_{key}.json",
#         f"/home/shuai.liu01/PaddleXrc/input_jsons/output_{key}.json",
#     )
num = 70
# direct_test(
#     f"/home/user/liushuai/DocBench-100/{num}/input_{num}.json",
#     f"/home/user/liushuai/DocBench-100/{num}/out_{num}.json",
# )
direct_test(
    f"/home/user/liushuai/DocBench-100/mineru/input_{num}.json",
    f"/home/user/liushuai/DocBench-100/mineru/out_{num}.json",
)
