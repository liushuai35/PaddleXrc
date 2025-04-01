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

from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="layout_parsing_v2", device="gpu:0")

output = pipeline.predict(
    "/home/shuai.liu01/DocBench-100/30/complex.pdf",
    # "/mnt/shuailiu35/eval_layout_order/70/input.pdf",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_common_ocr=True,
    use_seal_recognition=False,
    use_table_recognition=False,
    min_gap_x=1000,
    min_gap_y=1000,
    is_only_x=True,
    # page_test_index=22,
)

for res in output:
    # res.print()
    res.save_to_img("./complex_30")
    res.save_to_json("./complex_30")
