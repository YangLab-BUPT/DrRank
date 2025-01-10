import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import time


class LlmPointwiseRanker:
    def __init__(
        self, hf_model=None, tokenizer=None, ranking_criteria_map=None, label_nums=5, query_format="{}"
    ):
        super().__init__()

        self.model = hf_model
        self.tokenizer = tokenizer
        self.query_format = query_format
        self.ranking_criteria_map = ranking_criteria_map
        self.label_map = self.get_label_map(label_nums)
        labels_str = ", ".join([f'"{k}"' for k in self.label_map.keys()])
        print(f"Using label map: {self.label_map}")

        self.sys_prompt = "您是一位擅长根据患者需求进行医生推荐的智能医疗助手, 您具备丰富的医疗背景知识和出色的分析推理能力。"
        self.user_prompt = (
            "患者需求: <<{patient_need}>>\n候选医生: <<{candidate_doctor}>>\n\n请您根据医生信息判断候选医生在患者医疗需求下的专业权威程度, 将其自上而下分为"
            + f"{label_nums}级: "
            + f"{labels_str}。"
            + "\n{criteria}\n"
        )

        self.score_elicit = '候选医生在患者医疗需求下的专业权威程度为"'

        self.explanation_elicit = '"。\n\n理由如下：\n\n1. '
        self.explanation_elicit_token_ids = torch.tensor(
            self.tokenizer.encode(self.explanation_elicit)
        ).unsqueeze(0)

        self.label_token_id = {
            self.tokenizer.encode(k)[0]: k for k, v in self.label_map.items()
        }
        self.label_token_id_list = torch.tensor(
            list(self.label_token_id.keys()))
        assert (
            sum([len(self.tokenizer.encode(k)) !=
                1 for k, v in self.label_map.items()])
            == 0
        )

        self.score_time = []
        self.explanation_time = []

    def get_label_map(self, label_nums):
        assert 2 <= label_nums <= 5
        label_map = None
        if label_nums == 2:
            label_map = {"高级": 1, "无关": 0}
        elif label_nums == 3:
            label_map = {"高级": 2, "初级": 1, "无关": 0}
        elif label_nums == 4:
            label_map = {"高级": 3, "中级": 2, "初级": 1, "无关": 0}
        elif label_nums == 5:
            label_map = {"顶级": 4, "高级": 3, "中级": 2, "初级": 1, "无关": 0}

        # label_map["无关"] = -1
        return label_map

    def forward(self, batch):
        # Generate scores
        start_time = time.time()
        # torch.Size([2, 622, 152064])
        generated_scores = self.model(**batch).logits
        self.score_time.append(time.time() - start_time)

        # Extract scores for each label
        scores = [
            {
                label: generated_scores[i, -1, label_id].detach().cpu().item()
                for label_id, label in self.label_token_id.items()
            }
            for i in range(generated_scores.shape[0])
        ]

        # Select max label token ids
        selected_values = generated_scores[:, -1, self.label_token_id_list]
        self.label_token_id_list = self.label_token_id_list.to(
            selected_values.device)
        max_label_token_ids = torch.argmax(selected_values, dim=1)
        max_label_token_ids = self.label_token_id_list[max_label_token_ids][:, None]
        # Update batch with new input_ids and attention_mask
        self.update_tensors(batch, max_label_token_ids)

        # Repeat and add explanation elicit token ids
        explanation_elicit_token_ids = self.explanation_elicit_token_ids.repeat(
            batch.input_ids.shape[0], 1
        )
        self.update_tensors(batch, explanation_elicit_token_ids)

        # Generate explanations
        start_time = time.time()
        generated_explanations = self.model.generate(
            **batch,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            output_logits=True,
            return_dict_in_generate=True,
        ).sequences
        self.explanation_time.append(time.time() - start_time)
        # Extract and decode explanations
        explanations = [
            self.tokenizer.decode(
                generated_explanations[i][len(batch["input_ids"][i]) - 3:]
                .detach()
                .cpu(),
                skip_special_tokens=True,
            )
            for i in range(generated_explanations.shape[0])
        ]

        return scores, explanations

    def update_tensors(self, batch, new_tokens):
        new_tokens = new_tokens.to(batch["input_ids"].device)
        batch["input_ids"] = torch.cat(
            [batch["input_ids"], new_tokens], dim=-1)
        batch["attention_mask"] = torch.cat(
            [batch["attention_mask"], torch.ones_like(new_tokens)], dim=-1
        )

    @torch.no_grad()
    def compute_score(
        self, sentences_pairs, batch_size=12, max_length=2048, normalize=True
    ):
        """
        sentences_pairs=[[query,title],[query1,title1],...]
        """
        all_scores = []
        all_explanations = []
        for start_index in tqdm(range(0, len(sentences_pairs), batch_size)):
            sentences_batch = sentences_pairs[start_index: start_index + batch_size]
            batch_data, messages_batch = self.preprocess(
                sentences_batch, document_max_length=max_length
            )  # Some api endpoints need messages_batch
            batch_data = batch_data.to(self.model.device)

            scores, explanations = self.forward(batch_data)

            assert len(scores) == len(sentences_batch)
            assert len(explanations) == len(sentences_batch)

            all_scores.extend(scores)
            all_explanations.extend(explanations)

        assert len(all_scores) == len(sentences_pairs)
        return all_scores, all_explanations

    def preprocess(self, sentences, document_max_length):
        import re

        message_batch = []
        for query, document in sentences:
            new_query = self.query_format.format(query)
            document_invalid_ids = self.tokenizer.encode(
                document,
                max_length=document_max_length,
                truncation=True,
                add_special_tokens=False,
            )
            new_document = self.tokenizer.decode(document_invalid_ids)

            criteria = ""
            if self.ranking_criteria_map is not None:
                match = re.search(r"擅长使用(.*?)治疗(.*?)的医生专家", query)
                treatment = match.group(1)
                disease = match.group(2)
                key = f"{disease}-{treatment}"
                if key not in self.ranking_criteria_map:
                    print(f"Missing ranking criteria for {key}")
                    self.ranking_criteria_map[key] = "暂无评价标准"
                criteria = (
                    "您需要注意如下的判断标准: \n" + self.ranking_criteria_map[key]
                )

            prompt = self.user_prompt.format(
                patient_need=new_query.strip(),
                candidate_doctor=new_document.strip(),
                criteria=criteria.strip(),
            )
            message_batch.append(
                [
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": prompt},
                ]
            )

        text_batch = self.tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True,
        )
        text_batch = [text + self.score_elicit for text in text_batch]
        model_inputs_batch = self.tokenizer(
            text_batch, return_tensors="pt", padding="longest"
        )
        return model_inputs_batch, message_batch

    @classmethod
    def from_pretrained(
        cls, model_name_or_path, using_criteria=False, label_nums=5, query_format="{}"
    ):
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            trust_remote_code=True,
        )

        ranking_criteria_map = None
        # criteria_file = f"adapter/DrRank_criteria.{model_name_or_path.split('/')[-1]}.json"
        criteria_file = "adapter/DrRank_criteria.Qwen2.5-7B-Instruct.json"
        if using_criteria:
            with open(
                criteria_file,
                "r",
                encoding="utf-8",
            ) as f:
                ranking_criteria_map = json.load(f)
            print(f"Using ranking criteria from '{criteria_file}'")

        reranker = cls(hf_model, tokenizer, ranking_criteria_map,
                       label_nums, query_format)
        return reranker


def test_LlmPointwiseRanker(input_list):
    ckpt_path = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_reranker = LlmPointwiseRanker.from_pretrained(
        model_name_or_path=ckpt_path,
        using_criteria=True,
        label_nums=5,
    )
    all_score, all_analysis = llm_reranker.compute_score(input_list)
    print(f"{all_score=}")
    print(f"{all_analysis=}")


if __name__ == "__main__":
    input_list = [
        [
            "擅长使用药物治疗白血病的医生专家",
            "<MASK>是中国医学科学院血液病医院血液内科的一名主任医师。他的专业方向包括血液科。他擅长各种贫血、急慢性白血病、淋巴瘤、多发性骨髓瘤及疑难血液病、慢性粒细胞白血病等骨髓增殖性疾病。他的个人简介是：<MASK>，男，主任医师，教授，硕士生导师，政府特贴专家，1963年毕业于南京医学院。一直于中国医学科学院血液学研究所血液病医院血液内科工作。 现任天津医学会血液学分会副主任委员、中国医学科学院中国协和医科大学第五届学术委员会委员、《中华血液学杂志》副总编、《中国实用内科杂志》资深编委、《白血病与淋巴瘤》常委编委、《临床血液学杂志》、《肿瘤防治杂志》编委、中华医学会医疗事故技术鉴定专家库成员。1989年获天津市“七五”立功奖章；1997年获天津市“九五”立功先进个人称号；1992年享受政府特殊津贴；1997年获卫生部有突出贡献的中青年专家。曾承担和参加国家自然科学基金、卫生部基金、医科院基金等8项课题研究工作。发表学术论文180余篇，主编《血液内科主治医生450问》、《贫血》和参编《血液病实验诊断学》、《血液病学》等15部血液病专著。1981年、1995年两次获国家级发明三等奖，1991、2001、2003、2005年获卫生部天津市科学技术奖5项。 <MASK>教授从事医、教、研工作近43年，是国内知名的血液病专家。对于各种贫血、急慢性白血病、淋巴瘤、多发性骨髓瘤及疑难血液病的诊断治疗具有较全面的临床经验。尤其是慢性粒细胞白血病等骨髓增殖性疾病的中西医结合治疗及个体化治疗方面，更具有丰富的实践经验。 1996年以来，他坚持走中西医结合道路，深入研究慢性粒细胞白血病的中医治则，从采用中成药当归芦荟丸取得成功后，拆方研究，证实了其中的青黛及其靛玉红对慢性白血病的疗效。1978年至1979年他负责组织全国50家医院验证靛玉红对慢粒白血病的疗效。首创使用了双吲哚类药物治疗肿瘤的新方法。1983年后又负责研究靛玉红的衍生物——甲异靛全国性临床试验。目前甲异靛已成为国家级新的一类抗肿瘤西药。",
        ],
        [
            "擅长使用药物治疗白血病的医生专家",
            "<MASK>是秦皇岛市第一医院血液科的一名主任医师。他的专业方向包括血液科。他擅长白细胞减少、各类贫血、MDS、各类白血病、骨髓增殖性疾病、血小板减少紫癜等血液系统疾病。他的个人简介是：<MASK>，男，主任医师，1986年毕业于河北医学院医学系，1993年晋升为主治医师，1998年晋升为副主任医师。秦皇岛市第一医院血液科主任，1993~1994年在北京医科大学第一临床学院进修血液病专业1年，2000~2001年作为河北省中青年学科带头人在复旦大学附属中山医院研修血液病专业1年，2001年开始在河北医学院研究生班学习，现已经通过英语和医学综合的全国统考，并准备论文答辩而获得医学硕士。 十几年来，潜身钻研血液病的诊断和治疗并获得了巨大的成功，治愈了成千上万的疑难血液病人。在理论和实践的结合过程中不断总结经验，先后撰写了几十篇有价值的医学论文，尤其是1998年以来发表国家级医学论文11篇，主要包括：反应停诱导骨髓瘤细胞株（KM3）凋亡的初步研究、血液肿瘤学编者和译者并翻译完成成人急性髓细胞性白血病的诊断与治疗、白细胞介素-1抑制因子对急慢性髓细胞性白血病的治疗、白细胞的促凝血活性研究、反应停治疗多发性骨髓瘤的基础和临床研究、红细胞生成临床应用以及利福平致免疫性血小板减少、遗传因素与环境因素与白血病和骨髓增生异常综合症等。获得了医学同行的一致首肯和赞扬。为了更好的为病人以及全社会服务，今后一定更加努力学习和工作并希望得到同行和各位病人的支持指导和合作以便共同提高专业技术水平，并希望及时与我密切沟通。",
        ],
    ]
    test_LlmPointwiseRanker(input_list)
