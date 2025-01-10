import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tqdm
import time


class CrossEncoder:
    def __init__(
        self,
        hf_model=None,
        tokenizer=None,
        query_format="{}",
        document_format="{}",
    ):
        super().__init__()

        self.model = hf_model
        self.tokenizer = tokenizer
        self.query_format = query_format
        self.document_format = document_format
        
        self.score_time = []
        self.explanation_time = []

    def forward(self, batch):
        start_time = time.time()
        output = self.model(**batch)
        self.score_time.append(time.time() - start_time)
        return output

    @torch.no_grad()
    def compute_score(
        self, sentences_pairs, batch_size=256, max_length=512, normalize=False
    ):
        """
        sentences_pairs=[[query,title],[query1,title1],...]
        """

        all_logits = []
        for start_index in tqdm.tqdm(range(0, len(sentences_pairs), batch_size)):
            sentences_batch = sentences_pairs[start_index : start_index + batch_size]
            batch_data = self.preprocess(
                sentences_batch, document_max_length=max_length
            ).to(self.model.device)
            output = self.forward(batch_data)
            logits = output.logits.detach().cpu()
            all_logits.extend(logits)

        if normalize:
            all_logits = torch.sigmoid(torch.tensor(all_logits)).detach().cpu().tolist()

        return all_logits

    def preprocess(self, sentences_pairs, document_max_length):
        new_sentences_pairs = []
        for query, document in sentences_pairs:
            new_query = self.query_format.format(query.strip())
            new_document = self.tokenizer.decode(
                self.tokenizer.encode(self.document_format.format(document.strip()))[
                    :document_max_length
                ]
            )
            new_sentences_pairs.append([new_query, new_document])
        assert len(new_sentences_pairs) == len(sentences_pairs)

        tokens = self.tokenizer.batch_encode_plus(
            new_sentences_pairs,
            add_special_tokens=True,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        )
        return tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        num_labels=1,
        query_format="{}",
        document_format="{}",
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        reranker = cls(
            hf_model,
            tokenizer,
            query_format,
            document_format,
        )
        return reranker


def test_CrossEncoder(input_list):
    ckpt_path = "/data/zzy/Models/bge-reranker-v2-m3"
    reranker = CrossEncoder.from_pretrained(
        model_name_or_path=ckpt_path,
        num_labels=1,  # binary classification
    )

    all_score = reranker.compute_score(input_list, normalize=True)
    print(f"{all_score=}")


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
    test_CrossEncoder(input_list)
