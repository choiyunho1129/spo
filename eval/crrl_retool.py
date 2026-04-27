# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modifications Copyright 2025 CRRL authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

import datasets

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    @rollout_trace_op
    async def execute(self, instance_id: str, code: str, **kwargs) -> tuple[str, float, dict]:
        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        timeout = self.default_timeout
        language = self.default_language
        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        # sandbox has no score or metrics, use Nones
        return result, None, None


answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""

DAPO_SINGLE_TURN_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: $Answer (without quotes) "
    "where $Answer is the answer to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    @staticmethod
    def _normalize_data_source(path_or_name: str) -> str:
        source = path_or_name.replace("\\", "/")
        source_lower = source.lower()
        if "maxwell-jia/aime_2024" in source_lower or "aime_2024" in source_lower:
            return "Maxwell-Jia/AIME_2024"
        if "yentinglin/aime_2025" in source_lower or "aime_2025" in source_lower:
            return "yentinglin/aime_2025"
        if "bytedance-seed/beyondaime" in source_lower or "beyondaime" in source_lower:
            return "ByteDance-Seed/BeyondAIME"
        if "matharena/brumo_2025" in source_lower or "brumo_2025" in source_lower:
            return "MathArena/brumo_2025"
        if "matharena/hmmt_feb_2025" in source_lower or "hmmt_feb_2025" in source_lower:
            return "MathArena/hmmt_feb_2025"
        if "polaris-dataset-hard" in source_lower:
            return "dataset/Polaris-Dataset-Hard"
        if "open-r1/dapo-math-17k-processed" in source_lower:
            return "open-r1/DAPO-Math-17k-Processed"
        return "/".join(source.split("/")[-2:])

    def _build_dapo_source_prompt(self, problem: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": DAPO_SINGLE_TURN_PROMPT_TEMPLATE.format(problem=problem)}]

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            data_source = self._normalize_data_source(parquet_file)
            # read parquet files and cache
            if ".parquet" in parquet_file:
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            elif data_source == "open-r1/DAPO-Math-17k-Processed":
                dataframe = datasets.load_dataset(parquet_file, "all")["train"]
            elif data_source == "ByteDance-Seed/BeyondAIME":
                dataframe = datasets.load_dataset(parquet_file)["test"]
            elif data_source == "dataset/Polaris-Dataset-Hard":
                dataframe = datasets.load_from_disk(parquet_file)
            else:
                dataframe = datasets.load_dataset(parquet_file)["train"]
            if data_source in [
                "Maxwell-Jia/AIME_2024",
                "yentinglin/aime_2025",
                "ByteDance-Seed/BeyondAIME",
                "MathArena/brumo_2025",
                "MathArena/hmmt_feb_2025",
            ]:
                dataframe = dataframe.map(
                    self.map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
                )
            elif data_source == "dataset/Polaris-Dataset-Hard":
                dataframe = dataframe.map(
                    self.map_fn,
                    fn_kwargs={"data_source": "dataset/Polaris-Dataset-Hard"},
                    remove_columns=dataframe.column_names,
                )
            else:
                dataframe = dataframe.map(self.map_fn2, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict, *, data_source: str = None):
        if data_source == "Maxwell-Jia/AIME_2024":
            problem, answer = row["Problem"], row["Answer"]
        elif data_source in [
            "yentinglin/aime_2025",
            "ByteDance-Seed/BeyondAIME",
            "MathArena/brumo_2025",
            "MathArena/hmmt_feb_2025",
        ]:
            problem, answer = row["problem"], row["answer"]
        elif data_source == "dataset/Polaris-Dataset-Hard":
            problem, answer = row["problem"], row["answer"]

        data = {
            "data_source": data_source.split("/")[1].lower(),  # aime_2024, aime_2025, polaris-dataset-hard
            # "prompt": problem,
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
        }
        if self.prompt_key == "source_prompt":
            data["source_prompt"] = self._build_dapo_source_prompt(problem)
        else:
            data["prompt"] = [{"role": "user", "content": problem + answer_format}]
            data["agent_name"] = "crrl_tool_agent"
        return data

    def map_fn2(self, row: dict):
        if self.prompt_key == "source_prompt":
            source_prompt = row.get("source_prompt")
            if source_prompt is not None:
                row["source_prompt"] = list(source_prompt)
            else:
                row["source_prompt"] = self._build_dapo_source_prompt(row.get("prompt") or row.get("problem") or row.get("question"))
            row.pop("agent_name", None)
            return row
        print("prompt")
        exit()
        content = row.get("prompt") or row.get("problem") or row.get("question")
        row["prompt"] = [{"role": "user", "content": content + answer_format}]
        row["agent_name"] = "crrl_tool_agent"
        return row


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    # Check format: if more than one "</think>" tag, score should be zero
    if solution_str.count("</think>") != 1:
        return {"score": 0, "acc": False, "pred": ""}

    # Check if there are <code> or <interpreter> blocks after </think>
    think_end_pos = solution_str.find("</think>")
    if think_end_pos != -1:
        after_think = solution_str[think_end_pos + len("</think>") :]
        if "<code>" in after_think or "<interpreter>" in after_think:
            return {"score": 0, "acc": False, "pred": ""}

    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

    # Modify to 0, +1 reward
    if result["score"] < 0:
        result["score"] = 0

    if result["pred"] is None:
        result["pred"] = ""

    return result
