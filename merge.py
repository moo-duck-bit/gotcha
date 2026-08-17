#!/usr/bin/env python3
import os
import json
from glob import glob
from collections import defaultdict

def load_all_json_files_from_dir(directory):
    """주어진 디렉토리에서 모든 JSON 파일을 읽어 리스트로 반환."""
    json_data = []
    json_files = glob(os.path.join(directory, "*.json"))
    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                json_data.extend(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding {filepath}: {e}")
    return json_data

def merge_json_by_id(core_dir, distortions_dir, fine_grained_dir, output_dir):
    # 모든 JSON 파일 불러오기
    core_data = load_all_json_files_from_dir(core_dir)
    distortions_data = load_all_json_files_from_dir(distortions_dir)
    fine_grained_data = load_all_json_files_from_dir(fine_grained_dir)

    # 데이터를 id별로 정리
    id_data = defaultdict(dict)

    # core 데이터 처리
    for item in core_data:
        item_id = str(item.get("id")).strip()
        if not item_id:
            continue
        id_data[item_id].setdefault("ori_text", item.get("ori_text", ""))
        id_data[item_id].setdefault("thoughts", item.get("thoughts", ""))
        if "core_belief_major" in item:
            id_data[item_id]["core_belief"] = item["core_belief_major"]

    # fine_grained 데이터 처리
    for item in fine_grained_data:
        item_id = str(item.get("id")).strip()
        if not item_id:
            continue
        if "fine_grained" in item:
            id_data[item_id]["fine_grained"] = item["fine_grained"]

    # distortions 데이터 처리
    for item in distortions_data:
        item_id = str(item.get("id")).strip()
        if not item_id:
            continue
        if "distortions" in item:
            id_data[item_id]["distortions"] = item["distortions"]
        if "ori_text" in item and not id_data[item_id].get("ori_text"):
            id_data[item_id]["ori_text"] = item["ori_text"]
        if "thoughts" in item and not id_data[item_id].get("thoughts"):
            id_data[item_id]["thoughts"] = item["thoughts"]

    # 출력용 리스트 생성
    userinput_data = []
    answer_data = []

    for item_id, fields in sorted(id_data.items()):
        has_answer_data = any(
            key in fields for key in ["core_belief", "fine_grained", "distortions"]
        )
        if not has_answer_data:
            print(f"[경고] ID={item_id}에 answer data 없음 → 생략")
            continue

        # userinput
        userinput_data.append({
            "id": item_id,
            "ori_text": fields.get("ori_text", ""),
            "thoughts": fields.get("thoughts", "")
        })

        # answer
        answer_entry = {"id": item_id}
        if "core_belief" in fields:
            answer_entry["core_belief"] = fields["core_belief"]
        if "fine_grained" in fields:
            answer_entry["fine_grained"] = fields["fine_grained"]
        else:
            print(f"[정보] ID={item_id}에는 fine_grained 없음")
        if "distortions" in fields:
            answer_entry["distortions"] = fields["distortions"]

        answer_data.append(answer_entry)

    # JSON으로 저장
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "userinput.json"), "w", encoding="utf-8") as f:
        json.dump(userinput_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "answer.json"), "w", encoding="utf-8") as f:
        json.dump(answer_data, f, indent=2, ensure_ascii=False)

    print(f"\n총 {len(userinput_data)}개의 항목 저장 완료")
    print(f"- userinput.json: {os.path.join(output_dir, 'userinput.json')}")
    print(f"- answer.json: {os.path.join(output_dir, 'answer.json')}")

if __name__ == "__main__":
    core_dir = "/workspace/graph/CBT-Bench/core"
    fine_grained_dir = "/workspace/graph/CBT-Bench/fine"
    distortions_dir = "/workspace/graph/CBT-Bench/distortions"
    output_dir = "/workspace/graph"

    merge_json_by_id(core_dir, distortions_dir, fine_grained_dir, output_dir)
