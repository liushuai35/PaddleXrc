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

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from scipy.stats import kendalltau
import Levenshtein


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2: Lists or tuples representing bounding boxes [x_min, y_min, x_max, y_max].

    Returns:
        float: The IoU value.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0  # No intersection

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    min_area = min(box1_area, box2_area)

    iou = inter_area / min_area
    return iou


def get_minbox_if_overlap_by_ratio(bbox1, bbox2, ratio, smaller=True):
    """
    Determine if the overlap area between two bounding boxes exceeds a given ratio
    and return the smaller (or larger) bounding box based on the `smaller` flag.

    Args:
        bbox1, bbox2: Coordinates of bounding boxes [x_min, y_min, x_max, y_max].
        ratio (float): The overlap ratio threshold.
        smaller (bool): If True, return the smaller bounding box; otherwise, return the larger one.

    Returns:
        list or tuple: The selected bounding box or None if the overlap ratio is not exceeded.
    """
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    overlap_ratio = calculate_iou(bbox1, bbox2)

    if overlap_ratio > ratio:
        if (area1 <= area2 and smaller) or (area1 > area2 and not smaller):
            return bbox1
        else:
            return bbox2
    return None


def remove_overlaps_blocks(blocks, threshold=0.65, smaller=True):
    """
    Remove overlapping blocks based on a specified overlap ratio threshold.

    Args:
        blocks (list): List of block dictionaries, each containing a 'layout_bbox' key.
        threshold (float): Ratio threshold to determine significant overlap.
        smaller (bool): If True, the smaller block in overlap is removed.

    Returns:
        tuple: A tuple containing the updated list of blocks and a list of dropped blocks.
    """
    dropped_blocks = []
    for i, block1 in enumerate(blocks):
        for j, block2 in enumerate(blocks):
            if i >= j:
                continue
            if block1 in dropped_blocks or block2 in dropped_blocks:
                continue
            overlap_box = get_minbox_if_overlap_by_ratio(
                block1["layout_bbox"],
                block2["layout_bbox"],
                threshold,
                smaller=smaller,
            )
            if overlap_box:
                block_to_remove = next(
                    (
                        block
                        for block in (block1, block2)
                        if block["layout_bbox"] == overlap_box
                    ),
                    None,
                )
                if block_to_remove and block_to_remove not in dropped_blocks:
                    dropped_blocks.append(block_to_remove)

    for block in dropped_blocks:
        blocks.remove(block)
        block["tag"] = "block_overlap"
    return blocks, dropped_blocks


# def match_bboxes(input_bboxes, input_indices, gt_bboxes, gt_indices, iou_threshold=0.5):
#     """
#     Match input bounding boxes to ground truth bounding boxes based on IoU.

#     Args:
#         input_bboxes: List of input bounding boxes.
#         input_indices: List of input indices.
#         gt_bboxes: List of ground truth bounding boxes.
#         gt_indices: List of ground truth indices.
#         iou_threshold (float): IoU threshold for matching.

#     Returns:
#         tuple: Matched input indices and ground truth indices in relative order.
#     """
#     matched_pairs = []

#     # Step 1: Match input bboxes to gt bboxes
#     for input_idx, input_bbox in zip(input_indices, input_bboxes):
#         matched_gt_indices = []
#         for gt_idx, gt_bbox in zip(gt_indices, gt_bboxes):
#             iou = calculate_iou(input_bbox, gt_bbox)
#             if iou >= iou_threshold:
#                 matched_gt_indices.append(gt_idx)

#         # Process matches
#         if matched_gt_indices:
#             for gt_idx in matched_gt_indices:
#                 matched_pairs.append((input_idx, gt_idx))

#     # Step 2: Sort matched pairs by input indices
#     sorted_matches = sorted(matched_pairs, key=lambda x: (x[0], x[1]))

#     # Generate continuous indices for matched inputs
#     if not sorted_matches:
#         return [], []

#     matched_input = []
#     matched_gt = []
#     new_sorted_matches = []

#     for i, (x0, x1) in enumerate(sorted_matches):
#         new_sorted_matches.append((i + 1, x1))

#     new_sorted_matches.sort(key=lambda x: (x[1], x[0]))

#     for i, (x0, x1) in enumerate(new_sorted_matches):
#         matched_input.append(x0)
#         matched_gt.append(i + 1)

#     return matched_input, matched_gt


def match_bboxes(input_bboxes, input_indices, gt_bboxes, gt_indices, iou_threshold=0.5):
    """
    Match input bounding boxes to ground truth bounding boxes based on IoU.
    If no input bboxes match a gt bbox, use -1; otherwise, use the order of input bboxes.

    Args:
        input_bboxes: List of input bounding boxes.
        input_indices: List of input indices.
        gt_bboxes: List of ground truth bounding boxes.
        gt_indices: List of ground truth indices.
        iou_threshold (float): IoU threshold for matching.

    Returns:
        tuple: Matched input indices and ground truth indices in relative order, or -1 if no match.
    """
    if len(gt_indices) == 0:
        return [], []

    matched_pairs = []
    unmatched_gt_indices = set(gt_indices)
    gt_matched = [False] * (max(gt_indices) + 1)

    # Step 1: Match input bboxes to gt bboxes
    extra_num = 0
    for gt_idx, gt_bbox in zip(gt_indices, gt_bboxes):
        for input_idx, input_bbox in zip(input_indices, input_bboxes):
            iou = calculate_iou(input_bbox, gt_bbox)
            if iou >= iou_threshold:
                if gt_matched[gt_idx] == True:
                    extra_num += 1
                matched_pairs.append((input_idx, gt_idx + extra_num))
                unmatched_gt_indices.discard(gt_idx)
                gt_matched[gt_idx] = True

    # Handle unmatched gt bboxes
    unmatched_gt_indices = [index + extra_num for index in list(unmatched_gt_indices)]

    # Step 2: Sort matched pairs by input indices and handle multiple matches
    sorted_matches = sorted(matched_pairs, key=lambda x: x[0])

    # Generate continuous indices for matched inputs
    matched_input = []
    matched_gt = []
    new_sorted_matches = []

    for i, (x0, x1) in enumerate(sorted_matches):
        new_sorted_matches.append((i + 1, x1))

    new_sorted_matches.sort(key=lambda x: (x[1], x[0]))

    for i, (x0, x1) in enumerate(new_sorted_matches):
        matched_input.append(x0)
        # matched_gt.append(i + 1)
        matched_gt.append(x1)

    # Combine matched and unmatched indices
    final_input_indices = matched_input + [0] * len(unmatched_gt_indices)
    final_gt_indices = matched_gt + unmatched_gt_indices

    return final_input_indices, final_gt_indices


def calculate_metrics_with_block(
    block_index, input_bboxes, input_indices, gt_bboxes, gt_indices, debug=True
):
    """
    Calculate evaluation metrics (BLEU, ARD, TAU) for matched bounding boxes.

    Args:
        block_index: block index
        input_bboxes: List of input bounding boxes.
        input_indices: List of input indices.
        gt_bboxes: List of ground truth bounding boxes.
        gt_indices: List of ground truth indices.

    Returns:
        tuple: BLEU score, ARD, and TAU values.
    """
    sorted_matched_indices, sorted_gt_indices = match_bboxes(
        input_bboxes, input_indices, gt_bboxes, gt_indices, iou_threshold=0.5
    )
    if len(sorted_gt_indices) == 0:
        return 1, 0, 1, 0, 0

    if len(sorted_gt_indices) < 4 and sorted_gt_indices == sorted_matched_indices:
        bleu_score = 1
    else:
        # sorted_gt_indices = list(range(1,len(gt_bboxes) + 1))
        length = 4 - len(sorted_gt_indices)
        if length > 0:
            ext = list(range(len(sorted_gt_indices) + 1, 5))
        else:
            ext = []
        bleu_score = sentence_bleu(
            [sorted_gt_indices + ext], sorted_matched_indices + ext
        )  # references is list
        # if bleu_score<0.99 and length>0:
        #     print([sorted_gt_indices+ext], sorted_matched_indices+ext)
        #     print(bleu_score)
        #     raise ""

    if bleu_score < 0.99 and debug:
        print("block_index : ", block_index)
        print("bleu_score : ", bleu_score)
        print("input_bboxes : ", input_bboxes)
        print("gt_bboxes : ", gt_bboxes)
        print("sorted_matched_indices : ", sorted_matched_indices)
        print("sorted_gt_indices : ", sorted_gt_indices)
        print()

    if len(sorted_gt_indices) == 0:
        ard = 0
    else:
        ard = np.mean(
            [
                abs(pred - true) / true
                for pred, true in zip(sorted_matched_indices, sorted_gt_indices)
            ]
        )

    if sorted_matched_indices == sorted_gt_indices:
        tau = 1
    else:
        tau, _ = kendalltau(sorted_matched_indices, sorted_gt_indices)
        import math

        if math.isnan(tau):
            tau = 0

    edit_dist = Levenshtein.distance(sorted_matched_indices, sorted_gt_indices)

    return bleu_score, ard, tau, edit_dist, len(sorted_gt_indices)


def calculate_metrics_with_page(
    input_data, gt_data, iou_threshold=0.5, is_order_match=True, debug=True
):
    """
    Calculate evaluation metrics for pages, comparing input data to ground truth data.

    Args:
        input_data: List of input page data.
        gt_data: List of ground truth page data.
        iou_threshold (float): IoU threshold for matching.
        is_order_match (bool): If True, assumes ordered page matching.

    Returns:
        tuple: Averages of BLEU score, ARD, and TAU across pages.
    """
    assert len(input_data) == len(gt_data)
    total_bleu_score = 0
    total_ard = 0
    total_tau = 0
    total_match_block_num = 0
    total_edit_dist = 0
    total_length = 0

    if not is_order_match:
        for block in input_data:
            input_bbox = block["block_bbox"]
            for j, gt_block in enumerate(gt_data):
                gt_bbox = gt_block["block_bbox"]
                if calculate_iou(input_bbox, gt_bbox) > iou_threshold:
                    input_bboxes = block["sub_bboxes"]
                    input_indices = block["sub_indices"]
                    gt_bboxes = gt_block["sub_bboxes"]
                    gt_indices = gt_block["sub_indices"]
                    if 0 in input_indices:
                        input_indices = [index + 1 for index in input_indices]
                    if 0 in gt_indices:
                        gt_indices = [index + 1 for index in gt_indices]
                    bleu_score, ard, tau, edit_dist, length = (
                        calculate_metrics_with_block(
                            j,
                            input_bboxes,
                            input_indices,
                            gt_bboxes,
                            gt_indices,
                            debug,
                        )
                    )
                    total_bleu_score += bleu_score
                    total_ard += ard
                    total_tau += tau
                    total_edit_dist += edit_dist
                    total_match_block_num += 1
                    total_length += length
                    break
    else:
        bad_cases = []
        for block_index in range(len(input_data)):
            input_bboxes = input_data[block_index]["sub_bboxes"]
            gt_bboxes = gt_data[block_index]["sub_bboxes"]
            input_indices = input_data[block_index]["sub_indices"]
            gt_indices = gt_data[block_index]["sub_indices"]
            if 0 in input_indices:
                input_indices = [index + 1 for index in input_indices]
            if 0 in gt_indices:
                gt_indices = [index + 1 for index in gt_indices]
            bleu_score, ard, tau, edit_dist, length = calculate_metrics_with_block(
                block_index,
                input_bboxes,
                input_indices,
                gt_bboxes,
                gt_indices,
                debug,
            )
            if bleu_score < 0.95:
                bad_cases.append(block_index)
            total_bleu_score += bleu_score
            total_ard += ard
            total_tau += tau
            total_edit_dist += edit_dist
            total_match_block_num += 1
            total_length += length
        if debug:
            print("bad cases:", bad_cases)
    return (
        total_bleu_score / total_match_block_num,
        total_ard / total_match_block_num,
        total_tau / total_match_block_num,
        total_edit_dist / max(1, total_length),
    )


def paddlex_generate_input_data(data, gt_data=None):
    """
    Generate input data for evaluation based on layout parsing results.

    Args:
        data: Dictionary containing parsing results.

    Returns:
        list: Formatted list of input data.
    """
    parsing_result = data["parsing_res_list"]
    block_size = data["block_size"]
    gt_block_size = gt_data[0]["block_size"] if gt_data else block_size
    input_data = {
        "block_bbox": [0, 0, 2550, 2550],
        "sub_indices": [],
        "sub_bboxes": [],
        "page_scale": [
            gt_block_size[0] / block_size[0],
            gt_block_size[1] / block_size[1],
        ],
    }

    for sub_block in parsing_result:
        if sub_block.get("index") != None:
            input_data["sub_bboxes"].append(
                list(
                    map(
                        int,
                        np.array(sub_block["block_bbox"])
                        * np.array(input_data["page_scale"] * 2),
                    )
                )
            )
            input_data["sub_indices"].append(int(sub_block["index"]))

    return input_data


def get_gt_json(data, idx):
    """
    Generate input data for evaluation based on layout parsing results.

    Args:
        data: Dictionary containing parsing results.

    Returns:
        list: Formatted list of input data.
    """
    parsing_result = data["parsing_res_list"]
    block_size = data["block_size"]
    input_data = {
        "block_bbox": [0, 0, 2550, 2550],
        "sub_indices": [],
        "sub_bboxes": [],
        "sub_labels": [],
        "block_size": block_size,
        "page_idx": idx,
    }
    for sub_block in parsing_result:
        if sub_block.get("index") != None:
            input_data["sub_bboxes"].append(list(map(int, sub_block["block_bbox"])))
            input_data["sub_indices"].append(int(sub_block["index"]))
            input_data["sub_labels"].append(sub_block["block_label"])

    return input_data


def get_input_json(data):
    """
    Generate input data for evaluation based on layout parsing results.

    Args:
        data: Dictionary containing parsing results.

    Returns:
        list: Formatted list of input data.
    """
    parsing_result = data["parsing_res_list"]
    block_size = data["block_size"]
    for sub_block in parsing_result:
        sub_block["block_size"] = block_size
        del sub_block["index"]
        del sub_block["block_content"]

    # from paddlex.inference.pipelines.layout_parsing.utils import get_layout_ordering
    # parsing_result = get_layout_ordering(
    #     parsing_result,
    #     no_mask_labels=[
    #         "text",
    #         "formula",
    #         "algorithm",
    #         "reference",
    #         "content",
    #         "abstract",
    #     ],
    # )

    # for sub_block in parsing_result:
    #     sub_block["block_size"] = block_size
    return parsing_result


def mineru_generate_input_data(data, gt_data):
    """
    Generate input data for evaluation based on layout parsing results.

    Args:
        data: Dictionary containing parsing results.
        gt_data: Ground truth data for comparison.

    Returns:
        list: Formatted list of input data.
    """
    parsing_result = data["pdf_info"]
    parsing_result = [
        {
            "block_bbox": [0, 0, 2550, 2550],  # Page boundary bounding box
            "sub_blocks": block["para_blocks"],
            "block_size": block["page_size"],
        }
        for block in parsing_result
    ]

    input_data = [
        {
            "block_bbox": block["block_bbox"],
            "sub_indices": [],
            "sub_bboxes": [],
            "page_scale": [
                gt_block["block_size"][0] / block["block_size"][0],
                gt_block["block_size"][1] / block["block_size"][1],
            ],
        }
        for block, gt_block in zip(parsing_result, gt_data)
    ]

    for block_index, block in enumerate(parsing_result):
        sub_blocks = block["sub_blocks"]
        if len(sub_blocks) == 0:
            input_data[block_index]["sub_indices"] = [-1] * len(
                gt_data[block_index]["sub_indices"]
            )
        else:
            # sub_blocks = sorted(
            #     sub_blocks, key=lambda x: (x["index"], x.("bbox",[0,0])[1], x.get("bbox",[0,0])[0])
            # )

            for i, sub_block in enumerate(sub_blocks):
                if sub_block.get("index") != None:
                    input_data[block_index]["sub_bboxes"].append(
                        list(
                            map(
                                int,
                                np.array(sub_block["bbox"])
                                * np.array(input_data[block_index]["page_scale"] * 2),
                            )
                        )
                    )
                # input_data[block_index]["sub_indices"].append(i + 1)
                input_data[block_index]["sub_indices"].append(sub_block["index"])

    return input_data


def load_data_from_json(path):
    import json

    """
    Load data from a JSON file.

    Args:
        path (str): File path to the JSON file.

    Returns:
        dict: Parsed data from the JSON file.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def write_data_from_json(path, data):
    import json

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def main(
    debug=False, start_idx=0, end_idx=4, page_start_idx=0, page_end_idx=None, num=-1
):
    import json
    import os
    import glob

    total_bleu_score = 0
    total_ard = 0
    total_tau = 0
    total_edit_dist = 0
    keys = [
        "1andmore_column",
        "double_column",
        "three_column",
        "single_column",
    ]
    for index in range(start_idx, end_idx):
        dir_name = keys[index]

        if num != -1:
            input_json = f"/home/shuai.liu01/DocBench-100/{num}/out_{num}.json"
            gt_data = load_data_from_json(f"/home/shuai.liu01/gt_{num}.json")
        else:
            input_json = (
                f"/home/shuai.liu01/PaddleXrc/input_jsons/output_{dir_name}.json"
            )
            # input_json = (
            #     f"/home/shuai.liu01/PaddleXrc/mineru_results/output_{dir_name}.json"
            # )
            gt_data = load_data_from_json(
                f"/home/shuai.liu01/PaddleXrc/gt/gt_{dir_name}.json"
            )

        gt_data = gt_data[page_start_idx:page_end_idx]

        # PaddleX
        # input_jsons = glob.glob(
        #     f"/home/shuai.liu01/PaddleXrc/api_examples/pipelines/all_gt/{num}/*.json"
        # )
        # input_jsons = glob.glob(
        #     f"/home/shuai.liu01/PaddleXrc/api_examples/pipelines/{dir_name}/{num}/*.json"
        # )
        # input_jsons = glob.glob(
        #     # f"/home/shuai.liu01/PaddleXrc/api_examples/pipelines/{dir_name}/*.json"
        #     f"/home/shuai.liu01/PaddleXrc/api_examples/pipelines/complex_30/*.json"
        # )
        # input_jsons.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
        # input_data = []
        # for i, input_json in enumerate(input_jsons):
        #     if i == len(gt_data):
        #         break
        #     data = load_data_from_json(input_json)
        #     input_data.append(paddlex_generate_input_data(data, [gt_data[i]]))

        # input_data = load_data_from_json(input_json)

        input_data = []
        data = load_data_from_json(input_json)
        for i, page_data in enumerate(data):
            if i == len(gt_data):
                break
            input_data.append(paddlex_generate_input_data(page_data, [gt_data[i]]))

        # # MinerU
        # data = load_data_from_json(f"/home/shuai.liu01/PaddleXrc/mineru_results/mineru/{dir_name}_middle.json")
        # input_data = mineru_generate_input_data(data,gt_data)
        bleu_score, ard, tau, edit_dist = calculate_metrics_with_page(
            input_data, gt_data, debug=debug
        )
        print(
            f"{dir_name},BLEU score: {bleu_score}, ARD: {ard}, Tau :{tau}, Edit_dist:{edit_dist}"
        )
        total_bleu_score = total_bleu_score + bleu_score
        total_ard = total_ard + ard
        total_tau = total_tau + tau
        total_edit_dist = total_edit_dist + edit_dist
    return total_bleu_score, total_ard, total_tau, total_edit_dist


if __name__ == "__main__":
    main(True, start_idx=0, end_idx=4, num=30)
    # num_list = [30,70]
    # dir_list = ["all_gt"]

    # for num in num_list:
    #     for dir_name in dir_list:
    #         # PaddleX
    #         input_jsons = glob.glob(
    #             f"/home/shuai.liu01/PaddleXrc/api_examples/pipelines/{dir_name}/{num}/*.json"
    #         )

    #         input_jsons.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
    #         input_data = []
    #         for i, input_json in enumerate(input_jsons):
    #             # for generate gt json
    #             data = load_data_from_json(input_json)
    #             input_data.append(get_gt_json(data,i))

    #         write_data_from_json(f"/home/shuai.liu01/PaddleXrc/api_examples/pipelines/{dir_name}/gt_{num}.json",input_data)
