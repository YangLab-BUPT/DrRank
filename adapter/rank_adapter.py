from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
from openai import OpenAI
import threading
from concurrent.futures import ThreadPoolExecutor


class RankAdapter:
    def __init__(
        self,
        using_api=False,
        hf_model=None,
        tokenizer=None,
    ):
        super().__init__()
        self.model = hf_model
        self.tokenizer = tokenizer
        self.using_api = using_api

        if using_api:
            print("Using API endpoint!")
            self.client = OpenAI(api_key="sk-xxx", base_url="xxx")

        self.sys_prompt = "您是一位擅长分析医疗信息的智能助手，您具备丰富的医疗背景知识和出色的分析推理能力。"
        self.user_prompt = """您的任务是需要判断医生在<<{treatment}>>治疗<<{disease}>>领域下的专业权威性。\n您的判断是基于医生的个人信息，其中包括医生的个人简介、专业方向，教育经历、科研经历、获奖经历、社会任职等情况。\n您首先需要思考应该从哪些方面进行客观、公正地判断，并给出判断标准，过程如下: \n1.您需要分别从医生基本情况(以及所在医院的情况)、疾病相关性、治疗方式相关三个方面分析判断标准;\n2. 在各个方面中您需要尽可能从多角度给出判断标准;\n3. 您应该将判断标准分点回答，保证每一点的清晰、可执行、简洁等特性。\n
手术治疗胃癌的判断标准示例如下：
### 1. 医生基本情况

- **学历与资质**：医生是否拥有医学博士学位，以及是否持有国家认证的执业医师资格证和高级职称，如副主任医师或主任医师。
- **专业培训**：医生是否接受过专门针对胃癌的高级专业培训，如参加过国内外知名医疗机构的进修或研讨会。
- **学术成就**：医生在胃癌领域的学术贡献，如发表的高质量论文、参与的科研项目、获得的学术奖项等。
- **临床经验**：医生在胃癌手术治疗方面的实际操作经验，包括年手术量、成功案例数量及类型，特别是复杂病例的处理能力。
- **患者评价**：收集患者对医生手术技术、术后恢复指导、沟通能力等方面的评价，了解其在患者群体中的口碑。
- **所在医院**：医院在胃癌治疗领域的整体实力，如是否设有专门的胃癌诊疗中心，设备是否先进，科研投入是否充足，以及与其他医疗机构的合作情况。

### 2. 疾病相关性

- **研究贡献**：医生在胃癌领域的研究成果，如发表的学术论文、参与的科研项目、获得的专利等，特别是那些直接针对胃癌治疗的研究。
- **学术影响力**：通过引用次数、H指数等指标衡量医生在胃癌研究领域的学术影响力。
- **国际交流**：医生是否参与过国际胃癌学术会议，是否有与国际顶尖医疗机构或专家的合作经历。

### 3. 治疗方式相关

- **技术创新**：医生是否引入或应用了新的胃癌手术技术、治疗方法或器械，特别是在提高手术安全性、减少并发症、促进快速康复方面。
- **临床试验参与**：医生是否参与了胃癌相关的临床试验，特别是那些评估新疗法或手术方法有效性的试验。
- **教学与培训**：作为导师或讲师，医生在培训年轻医生或分享胃癌手术技巧方面的贡献，体现了其在专业领域的领导力和影响力。
- **患者教育**：医生是否积极参与胃癌患者的教育工作，提供术前术后指导，帮助患者更好地理解病情和治疗方案。
"""

    def forward(self, batch):
        generated_ids_batch = self.model.generate(
            **batch,
            max_new_tokens=1024,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )

        return generated_ids_batch

    def get_response_from_api_endpoint(self, messages, model="gpt-4o-2024-05-13"):
        import time
        import hashlib

        assert self.using_api == True, "API is not enabled!"

        def string_to_md5(string):
            md5 = hashlib.md5()
            md5.update(string.encode("utf-8"))
            return md5.hexdigest()

        cnt = 0
        openai_cache = dict()
        with open(
            ".cache/llm.jsonl",
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = json.loads(line)
                openai_cache.update(line)
        while True:
            try:
                key = model + "_" + json.dumps(messages, ensure_ascii=False)
                key = string_to_md5(key)
                if key in openai_cache:
                    response = openai_cache[key]
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=False,
                        temperature=0.01,
                    )
                    openai_cache[key] = response
                    with open(
                        ".cache/llm.jsonl",
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(json.dumps({key: response}, ensure_ascii=False) + "\n")

                # print(f"OpenAI response: {response}")
                response = response.choices[0].message.content
                return response
            except Exception as e:
                cnt += 1
                time.sleep(1)
                print(f"OpenAI request failed: {e}, retrying...{cnt}")

    def generate_ranking_criteria(self, pairs, batch_size=12):
        """
        pairs=[[disease1,treatment1], [disease2,treatment2], ...]
        """

        if self.using_api:
            _, messages = self.preprocess(pairs)
            with ThreadPoolExecutor(
                max_workers=batch_size,
                thread_name_prefix=f"pid={threading.get_ident()}_api_call",
            ) as executor:
                model_responses = list(
                    executor.map(self.get_response_from_api_endpoint, messages)
                )
        else:
            model_responses = []
            for start_index in tqdm(range(0, len(pairs), batch_size)):
                pairs_batch = pairs[start_index : start_index + batch_size]
                batch_data, _ = self.preprocess(pairs_batch)
                batch_data = batch_data.to(self.model.device)
                generated_ids_batch = self.forward(batch_data)
                generate_sequence_ids = generated_ids_batch.sequences[
                    :, batch_data["input_ids"].shape[-1] :
                ]
                batch_model_responses = self.tokenizer.batch_decode(
                    generate_sequence_ids, skip_special_tokens=True
                )
                model_responses.extend(batch_model_responses)

        assert len(model_responses) == len(pairs)
        return model_responses

    def preprocess(self, pairs_batch):
        message_batch = []
        for disease, treatment in pairs_batch:
            user_prompt = self.user_prompt.format(disease=disease, treatment=treatment)
            message_batch.append(
                [
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

        if self.using_api:
            return None, message_batch

        text_batch = self.tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs_batch = self.tokenizer(
            text_batch, return_tensors="pt", padding="longest"
        )

        return model_inputs_batch, message_batch

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        using_api=False,
    ):
        if not using_api:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                padding_side="left",
                trust_remote_code=True,
            )
        else:
            hf_model = None
            tokenizer = None

        adapter = cls(using_api, hf_model, tokenizer)
        return adapter


def run_rank_adapter():
    import re
    base_path = ""
    model_path = base_path + "Qwen/Qwen2.5-7B-Instruct"
    dataset_path = "data/DrRank_V2.jsonl"
    output_path = f"adapter/DrRank_V2_criteria.{model_path.split('/')[-1]}.json"
    
    pattern = r"擅长使用(.*?)治疗(.*?)的医生专家"
    rank_adapter = RankAdapter.from_pretrained(model_name_or_path=model_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line)["query"] for line in f]

    pairs = set()
    for query in queries:
        match = re.search(pattern, query)
        if match:
            treatment = match.group(1)
            disease = match.group(2)
            pairs.add((disease, treatment))

    pairs = list(pairs)
    print(f"Total pairs: {len(pairs)}")
    all_criteria = rank_adapter.generate_ranking_criteria(pairs, batch_size=12)

    adict = dict()
    for i, (disease, treatment) in enumerate(pairs):
        adict[f"{disease}-{treatment}"] = all_criteria[i]

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(adict, f, ensure_ascii=False, indent=4)

    print(pairs[0])
    print(all_criteria[0])


if __name__ == "__main__":
    run_rank_adapter()
