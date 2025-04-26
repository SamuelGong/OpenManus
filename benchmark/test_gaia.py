import os
import json
from run_gaia import GAIALoader
from scorer import question_scorer


def main():
    set_type = "validation"  # cannot be "test"
    input_file = f"gaia_{set_type}.jsonl"
    output_file = f"gaia_{set_type}_scored.jsonl"

    existing_result = {}
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    existing_result[record["task_id"]] = record
                except json.JSONDecodeError:
                    continue  # Skip any malformed lines
    print(f"{len(existing_result)} records collected.")

    score_dict = {}
    test_result = {}
    total_all = 0
    total_correct = 0
    for level in ["level1", "level2", "level3"]:
        print(f"Processing {level}")
        test_result[level] = {
            "raw": {}
        }
        gaia = GAIALoader(level)

        level_all = 0
        level_correct = 0
        task_list = gaia.dataset[set_type]
        for task in task_list:
            task_id = task.get("task_id")
            ground_truth = task["Final answer"]

            res = {
                "task_question": task["Question"],
                "file_name": task["file_name"],
                "ground_truth": ground_truth
            }
            if task_id not in existing_result:
                level_all += 1
                test_result[level]["raw"][task_id] = res
                print(f"\tResult of {task_id} not found.")
                continue

            model_answer = existing_result[task_id]["model_answer"]
            correct = question_scorer(ground_truth, model_answer)
            level_all += 1
            if correct:
                level_correct += 1
            res.update({
                "model_answer": model_answer,
                "correct": correct,
            })
            test_result[level]["raw"][task_id] = res

        level_result = {
            "total": level_all,
            "correct": level_correct,
            "score": round(level_correct / level_all, 2)
        }
        score_dict[level] = level_result
        test_result[level]["stat"] = level_result
        total_all += level_all
        total_correct += level_correct

    test_result["all_stat"] = {
        "total": total_all,
        "correct": total_correct,
        "score": round(total_correct / total_all, 2)
    }
    with open(output_file, "w") as fout:
        json.dump(test_result, fout, indent=4)

    print(f"\nLevel 1: {test_result['level1']['stat']}")
    print(f"Level 2: {test_result['level2']['stat']}")
    print(f"Level 3: {test_result['level3']['stat']}")
    print(f"All: {test_result['all_stat']}")


if __name__ == '__main__':
    main()
