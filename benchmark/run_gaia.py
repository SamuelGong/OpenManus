import os
import sys
import json
import time
import asyncio
import traceback
import logging
from datasets import load_dataset

sys.path.append('..')  # adjust it if necessary
from app.agent.manus import Manus
from app.logger import logger


class GAIALoader:
    def __init__(self, level):
        # level: level1, level2, level3, all
        self.dataset = load_dataset(
            path="gaia-benchmark/GAIA",
            name=f"2023_{level}",
            trust_remote_code=True
        )

    def task2query(self, task, output_file):
        query = 'Your task is: {}'.format(task['Question'])
        if task['file_name']:
            query = query + (f'\n{task["file_path"]} is the absolute file path you need to use.')

        output_file = os.path.abspath(output_file)
        query += (f'\nWrite down your answer to file {output_file} with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. '
                  f'YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. '
                  f"If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. "
                  f"If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. "
                  f"If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.")

        return query


async def call_agent(query, log_path):
    log_path = os.path.abspath(log_path)
    sink_id = logger.add(
        log_path,
        level="INFO",
        mode="a",
        format=(
            "[{level}]"  # log level
            "[{time:YYYY-MM-DD HH:mm:ss.SSS}]"  # timestamp + milliseconds
            "[{process}]"  # OS process ID
            "[{file.name}:{line}]: "  # source filename and line #
            "{message}"  # the log message
        ),
        encoding="utf-8",
        rotation=None
    )  # Add a per-run file sink using your desired log‐formatter
    agent = await Manus.create(do_benchmark=True)  # Zhifeng

    start_time = time.perf_counter()
    logger.info(f"Starting serving the query: {query}")
    try:
        await agent.run(query)

        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"Query served in {round(duration, 2)}s.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    except Exception as e:

        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.error(f"Failed to serve query due to {e} "
                     f"after {round(duration, 2)}s\n{traceback.format_exc()}")
    finally:
        await agent.cleanup()
        logger.complete()
        # Remove this sink so the next call creates a fresh log file
        logger.remove(sink_id)


async def main():
    # set_type = "test"  # or "validation"
    set_type = "validation"
    result_file = f"gaia_{set_type}.jsonl"
    retry_limit = 3

    # Load processed task_ids if result_file already exists.
    processed_tasks = set()
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_tasks.add(record["task_id"])
                except json.JSONDecodeError:
                    continue  # Skip any malformed lines

    # Process tasks and update result file incrementally.
    with open(result_file, 'a') as out_file:
        for level in ["level1", "level2", "level3"]:
            print(f"Processing {level}")
            gaia = GAIALoader(level)
            task_list = gaia.dataset[set_type]

            task_list_len = len(task_list)
            for task_idx, task in enumerate(task_list):
                task_id = task.get("task_id")

                if task_id in processed_tasks:
                    print(f"\t({task_idx + 1}/{task_list_len}) "
                          f"Skipping task {task_id} (already processed).")
                    continue
                print(f"\t({task_idx + 1}/{task_list_len}) "
                      f"Processing task {task_id}")
                log_dir = os.path.join('logs', f"{set_type}-{level}")
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, f"{task_id}.log")
                output_path = os.path.join(log_dir, f"{task_id}.txt")
                query = gaia.task2query(task, output_path)

                retry_time = 0
                final_answer = ""
                while retry_time < retry_limit:
                    retry_time += 1

                    start_time = time.perf_counter()
                    try:
                        await call_agent(query, log_path)
                    except Exception as e:
                        print(f"Retrying task {task_id} for the {retry_time}th time due to error {e}")
                        continue

                    if not os.path.exists(output_path):
                        print(f"Retrying task {task_id} for the {retry_time}th time "
                              f"as no output file generated")
                        continue
                    else:
                        with open(output_path, 'r') as f:
                            final_answer = f.read()
                        if "FINAL ANSWER: " in final_answer:
                            final_answer = final_answer.split("FINAL ANSWER: ")[1]

                        if not final_answer:
                            print(f"Retrying task {task_id} for the {retry_time}th time "
                                  f"as final answer cannot be found in the output file")
                            continue
                    break

                end_time = time.perf_counter()
                duration = round(end_time - start_time, 3)

                record = {"task_id": task_id, "model_answer": final_answer}
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

                processed_tasks.add(task_id)
                print(f"\tProcessed task {task_id} in {duration}s.")


if __name__ == "__main__":
    asyncio.run(main())
