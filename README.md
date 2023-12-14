# YAYI UIE大模型

<div align="center">
<img src="./assets/yayi_dark_small.png" alt="YaYi" style="width: 30%; display: block; margin: auto;">
<br>

[![License](https://img.shields.io/badge/Code%20License-Apache_2.0-brightgreen.svg)](./LICENSE)

[[📖README](./README.md)] 
[[🤗HF Repo](https://huggingface.co/wenge-research)]
[[🔗网页端](https://yayi.wenge.com)]

</div>


## 介绍
雅意信息抽取统一大模型 (YAYI-UIE)在百万级人工构造的高质量信息抽取数据上进行指令微调，统一训练信息抽取任务包括命名实体识别（NER），关系抽取（RE）和事件抽取（EE），实现通用、安全、金融、生物、医疗、商业、个人、车辆、电影、工业、餐厅、科学等场景下结构化抽取。

通过雅意IE大模型的开源为促进中文预训练大模型开源社区的发展，贡献自己的一份力量，通过开源，与每一位合作伙伴共建雅意大模型生态。

![instruction](/assets/YAYI-UIE-1.png)

## 模型地址
| Model Name | 🤗HF |  Download Links  |
| --------- | ---------    | --------- |
|  YAYI-UIE  | wenge-research/yayi-uie  | [模型下载](https://huggingface.co/wenge-research/yayi-uie)  |


## 训练数据
百万级语料中文54%，英文46%；其中数据集包括12个领域包括金融，社会，生物，商业，工业制造，化学，车辆，科学，疾病医疗，个人生活，安全和通用。覆盖数百个场景
- NER：中文覆盖**28**个实体类型包括人物，地缘政治，组织，身体部位，药物等，英文覆盖**130**个实体类型包括Animal, Weapon, Conference, Book等。
- RE：中文覆盖**232**种关系包括买资，增持，重组，国籍，别名，亲属，入股，转让，导致，发生地点，制造商等，英文覆盖**236**种关系包括founded by，state or province of headquarters，employee of，occupation，creator等。
- EE：中文覆盖**84**种事件类型,包括中标，高管变动，产品行为-发布，公司上市等，和**203**种论元，英文覆盖**45**种事件类型，包括Born, Demonstrate, Meet, End Organization, Divorce等，和**62**种论元。

![数据分布](/assets/data-dist.png)

## 运行方式
#### 安装环境
1. 下载本仓库内容至本地/远程服务器

```bash
git clone https://github.com/wenge-research/yayi-uie.git
cd yayi-uie
```

2. 创建conda环境

```bash
conda create --name uie python=3.8
conda activate uie
```

3. 安装环境

```bash
pip install -r requirements.txt
```
其中 `torch` 和 `transformers` 版本不建议低于推荐版本。

#### 模型推理
模型已在我们的 [Huggingface 模型仓库](https://huggingface.co/wenge-research) 开源，欢迎下载使用。以下是一个简单调用 `YAYI-UIE` 进行下游任务推理的示例代码，可在单张 A100/A800 等GPU运行，使用FP16精度推理时约占用 33GB 显存：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("wenge-research/yayi-uie", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("wenge-research/yayi-uie", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained("wenge-research/yayi-uie")
prompt = "文本:氧化锆陶瓷以其卓越的物理和化学特性在多个行业中发挥着关键作用。这种材料因其高强度、高硬度和优异的耐磨性，广泛应用于医疗器械、切削工具、磨具以及高端珠宝制品。在制造这种高性能陶瓷时，必须遵循严格的制造标准，以确保其最终性能。这些标准涵盖了从原材料选择到成品加工的全过程，保障产品的一致性和可靠性。氧化锆的制造过程通常包括粉末合成、成型、烧结和后处理等步骤。原材料通常是高纯度的氧化锆粉末，通过精确控制的烧结工艺，这些粉末被转化成具有特定微观结构的坚硬陶瓷。这种独特的微观结构赋予氧化锆陶瓷其显著的抗断裂韧性和耐腐蚀性。此外，氧化锆陶瓷的热膨胀系数与铁类似，使其在高温应用中展现出良好的热稳定性。因此，氧化锆陶瓷不仅在工业领域，也在日常生活中的应用日益增多，成为现代材料科学中的一个重要分支。\n抽取文本中可能存在的实体，并以json{制造品名称/制造过程/制造材料/工艺参数/应用/生物医学/工程特性：[实体]}格式输出。"
# "<reserved_13>" is a reserved token for human, "<reserved_14>" is a reserved token for assistant
prompt = "<reserved_13>" + prompt + "<reserved_14>"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
response = model.generate(**inputs, max_new_tokens=512, temperature=0)
print(tokenizer.decode(response[0],skip_special_tokens=True))
```

#### 指令样例
注：
- 指令前加入具体任务类型用中括号表示【】（可加可不加）
- 为了让模型能抽取更全的信息，尽量在指令中加入细粒度的提示，比如“会见地点”，“会议地点”等，而不是统一为“地点”。
- 尽量输入文本放置在前，指令在后。


1. 实体抽取任务
```
文本：xx
【实体抽取】抽取文本中可能存在的实体，并以json{人物/机构/地点：[实体]}格式输出。
```
2. 关系抽取任务
```
文本：xx
【关系抽取】已知关系列表是[注资,拥有,纠纷,自己,增持,重组,买资,签约,持股,交易]。根据关系列表抽取关系三元组，按照json[{'relation':'', 'head':'', 'tail':''}, ]的格式输出。
```
```
文本：xx
抽取文本中可能存在的关系，并以json[{'关系':'会见/出席', '头实体':'', '尾实体':''}, ]格式输出。
```
3. 事件抽取任务
```
文本：xx
已知论元角色列表是[质押方,披露时间,质权方,质押物,质押股票/股份数量,事件时间,质押物所属公司,质押物占总股比,质押物占持股比]，请根据论元角色列表从给定的输入中抽取可能的论元，以json{角色:论元,}格式输出。
```
```
文本：xx
已知论元角色列表是[时间，地点，会见主体，会见对象]，请根据论元角色列表从给定的输入中抽取可能的论元，以json{角色:论元}格式输出。
```

## 模型zero-shot评测
1. NER任务

AI，Literature，Music，Politics，Science为英文数据集，boson，clue，weibo为中文数据集

| 模型 | AI | Literature | Music | Politics | Science | 英文平均 | boson | clue | weibo | 中文平均 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| davinci | 2.97 | 9.87 | 13.83 | 18.42 | 10.04 | 11.03 | - | - | - | 31.09 |
| ChatGPT 3.5 | **54.4** | **54.07** | **61.24** | **59.12** | **63** | **58.37** | 38.53 | 25.44 | 29.3 |
| UIE | 31.14 | 38.97 | 33.91 | 46.28 | 41.56 | 38.37 | 40.64 | 34.91 | 40.79 | 38.78 |
| USM | 28.18 | 56 | 44.93| 36.1 | 44.09 | 41.86 | - | - | - | - |
| InstructUIE |	49 | 47.21 | 53.16 | 48.15 | 49.3 | 49.36 | - | - | - | - |
| DeepKE-LLM | 13.76 | 20.18 | 14.78 | 33.86 | 9.19 | 18.35 | 25.96 | 4.44 | 25.2 | 18.53 |
| YAYI-UIE | 52.4 | 45.99 | 51.2	| 51.82 | 50.53 | 50.39 | **49.25** | **36.46** | 36.78 | **40.83** |

2. RE任务

FewRe，Wiki-ZSL为英文数据集， SKE 2020，COAE2016，IPRE为中文数据集

| 模型 | FewRel | Wiki-ZSL | 英文平均 | SKE 2020 | COAE2016 | IPRE | 中文平均 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| ChatGPT 3.5 | 9.96 | 13.14 | 11.55  24.47 | 19.31 | 6.73 | 16.84 |
| ZETT(T5-small) | 30.53 | 31.74 | 31.14 | - | - | - | - |
| ZETT(T5-base) | 33.71 | 31.17 | 32.44 | - | - | - | - |
| InstructUIE |**39.55** | 35.2 | 37.38 | - | - | - | - |
| DeepKE-LLM | 17.46 | 15.33 | 16.40 | 0.4 | 6.56 | 9.75 |5.57|
| YAYI-UIE | 36.09 | **41.07** | **38.58** | **70.8** | **19.97** | **22.97**| **37.91**|

3. EE任务

commodity news为英文数据集，FewFC，ccf_law为中文数据集

EET（事件类型判别）

| 模型 | commodity news | FewFC | ccf_law | 中文平均 |
| ------ | ------ | ------ | ------ | ------ |
| ChatGPT 3.5 | 1.41 | 16.15 | 0 | 8.08 |
| UIE | - | 50.23 | 2.16 | 26.20 |
|InstructUIE| **23.26** | - | - | - |
| YAYI-UIE | 12.45 | **81.28** | **12.87** | **47.08**|

EEA（事件论元抽取）

| 模型 | commodity news | FewFC | ccf_law | 中文平均 |
| ------ | ------ | ------ | ------ | ------ |
| ChatGPT 3.5 | 8.6 | 44.4 | 44.57 | 44.49 |
| UIE | - | 43.02 | **60.85** | 51.94 |
|InstructUIE| **21.78** | - | - | - |
| YAYI-UIE | 19.74 | **63.06** | 59.42 | **61.24** |


![零样本推理性能分布](/assets/zh-0shot.png)

## 相关协议
#### 局限性
基于当前数据和基础模型训练得到的SFT模型，在效果上仍存在以下问题：

1. 抽取的信息可能会产生违背事实的错误回答。
2. 对于具备危害性的指令无法很好的鉴别，可能会产生危害性言论。
3. 在一些涉及段落级长文本的场景下模型的抽取能力仍有待提高。


#### 免责声明
基于以上模型局限性，我们要求开发者仅将我们开源的代码、数据、模型及后续用此项目生成的衍生物用于研究目的，不得用于商业用途，以及其他会对社会带来危害的用途。请谨慎鉴别和使用雅意大模型生成的内容，请勿将生成的有害内容传播至互联网。若产生不良后果，由传播者自负。
本项目仅可应用于研究目的，项目开发者不承担任何因使用本项目（包含但不限于数据、模型、代码等）导致的危害或损失。详细请参考免责声明。

#### 开源协议
本项目中的代码和数据依照 [Apache-2.0](LICENSE) 协议开源，社区使用YAYI UIE模型或其衍生品请遵循[Baichuan2](https://github.com/baichuan-inc/Baichuan2)的社区协议和商用协议。

## 更新日志
- [2023/12/15] YAYI-UIE大模型正式对外发布并开源。

## 致谢
- 本项目训练代码参考了 Databricks 的 [dolly](https://github.com/databrickslabs/dolly) 项目及 Huggingface [transformers](https://github.com/huggingface/transformers) 库；
- 本项目分布式训练使用了 Microsoft 的 [DeepSpeed](https://github.com/microsoft/deepspeed) 分布式训练工具及 Huggingface transformers 文档中的 [ZeRO stage 2](https://huggingface.co/docs/transformers/main_classes/deepspeed#zero2-config) 配置文件；
- 我们非常感谢以下开源项目对我们的帮助：[InstructUIE](https://github.com/BeyonderXX/InstructUIE/tree/master); [Baichuan2](https://github.com/baichuan-inc/Baichuan2); [InstructIE](https://github.com/zjunlp/DeepKE/tree/main/example/llm/InstructKGC); [DeepKE-LLM](https://github.com/zjunlp/KnowLM/tree/main)

## 引用
如果您在您的工作中使用了我们的模型，可以引用我们的论文：

```
@article{YAYI-UIE,
  author    = {Xinglin Xiao, Yijie Wang, Nan Xu, Yuqi Wang, Hanxuan Yang, Mingzheng Wang, Yin Luo, Lei Wang, Wenji Mao, Dajun Zeng}},
  title     = { YAYI-UIE: A Chat-Enhanced Instruction Tuning Framework for Universal Information Extraction},
  journal   = {arXiv preprint arXiv},
  year      = {2023}
}
```
