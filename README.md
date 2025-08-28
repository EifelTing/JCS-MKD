## The source code for "Joint class attention knowledge and self-knowledge for multi-teacher knowledge distillation" accepted by Engineering Applications of Artificial Intelligence (EAAI) 2025.
## Paper link: https://doi.org/10.1016/j.engappai.2025.111922

## Comparison

![alt text](https://github.com/EifelTing/JCS-MKD/blob/main/fig1.png)

## Abstract

Intelligent applications using large-scale deep neural networks face significant challenges due to their high
storage and computational demands, hindering deployment on resource-limited edge devices. Knowledge
distillation addresses this by transferring knowledge from an extensive teacher network to a smaller student
network, thereby reducing computational costs while preserving performance. Multi-teacher Knowledge Distil-
lation (MKD) further enhances this by allowing the student to learn from multiple teachers. However, MKD
methods have two key limitations: (1) They typically use non-interpretable logits or features as knowledge,
limiting the transparency of the learning process. (2) They focus primarily on teacher-guided learning, neglecting
the potential of combining teacher supervision with self-learning. To address these limitations, this study pre-
sents a novel method, Joint Class attention knowledge and Self-knowledge for Multi-teacher Knowledge Distillation
(JCS-MKD), which combines both teacher supervision and self-learning. Our method introduces two key in-
novations: (1) A class attention mechanism that integrates class activation maps from multiple teachers to deliver
more interpretable knowledge to the student. Additionally, an adaptive weighting scheme is employed to assign
greater importance to teacher predictions that are closer to the ground truth, ensuring the student primarily
learns from high-quality teacher knowledge. (2) A self-knowledge mechanism that decouples the student’s logit
into target and non-target components, customizing soft labels respectively to achieve adaptive self-supervision,
enabling the student to refine their understanding independently. Experimental results on standard benchmark
datasets demonstrate that JCS-MKD consistently outperforms state-of-the-art distillation methods across various
teacher-student architectures. 

## Method

![alt text](https://github.com/EifelTing/JCS-MKD/blob/main/fig2.png)

## Quick Start

### Reproduce our results
* 
    ```bash
    python tools/train.py --cfg configs/cifar100/JCS_MKD/res32x4_res8x4.yaml
    ```

## Citation:
  ```
@inproceedings{nayer,
  title={Joint class attention knowledge and self-knowledge for multi-teacher knowledge distillation},
  author={Ding Yifeng, Yang Gaoming, Ye Xinxin, Wang xiujun, Liu zhi},
  booktitle={Engineering Applications of Artificial Intelligence},
  vol={160},
  pages={111922},
  year={2025}
}
  ```
