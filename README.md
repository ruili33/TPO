# Temporal Preference Optimization (TPO) for Long-Form Video Understanding

<a href='https://arxiv.org/abs/2501.13919v1'><img src='https://img.shields.io/badge/arXiv-paper-red'></a> <a href='https://ruili33.github.io/tpo_website/'><img src='https://img.shields.io/badge/project-TPO-blue'></a> <a href='https://huggingface.co/collections/ruili0/temporal-preference-optimization-67874b451f65db189fa35e10'><img src='https://img.shields.io/badge/huggingface-datasets-green'></a> <a href='https://huggingface.co/collections/ruili0/temporal-preference-optimization-67874b451f65db189fa35e10'><img src='https://img.shields.io/badge/model-checkpoints-yellow'></a> 


<img src="asset/cvpr_figure_TPO.png"></img>
Our work proposes Temporal Preference Optimization (TPO), which serves as comprehensive pipeline for self-training based temporal preference optimization for cutting-edge video large multimodal models (video-LMMs). TPO enhances video comprehension in video-LMMs by modeling temporal preferences at two granular levels: localized and comprehensive TPO. In localized TPO (upper-left), we generate queries focused on short segments, with contrastive responses that retain or exclude the target segment. For comprehensive TPO (lower-left),  queries are designed broader understanding, using intact video versus sparse downsampled video for contrasting responses. After post-filtering, the contrast response pairs are serving as the  preference dataset to train a video-LMM, guiding the model to prioritize preferred responses for improved video understanding.


## :rocket: Quick Start

### Model Weights

| Model | Huggingface Link  |
:--------------------------:| :--------------------------:|
| LongVA-7B-TPO | [Download](https://huggingface.co/ruili0/LongVA-7B-TPO) |  
| LLaVA-Video-7B-TPO | [Download](https://huggingface.co/ruili0/LLaVA-Video-7B-Qwen2-TPO) |


### Install (Linux)
For  LongVA-TPO:
```
git clone https://github.com/ruili33/TPO
cd TPO
conda create -n TPOLongVA python=3.10
conda activate TPOLongVA
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e "longva/.[train]"
pip install packaging &&  pip install ninja && pip install flash-attn==2.5.0 --no-build-isolation --no-cache-dir
pip install -r requirements_longva.txt
```

For LLaVA-Video-TPO:
```
conda create -n TPOllava python=3.10 -y
conda activate TPOllava
pip install --upgrade pip 
pip install -e "LLaVA/.[train]"
pip install flash-attn==2.5.0 --no-build-isolation --no-cache-dir
```

### Inference

For LongVA-TPO, please following the inference demo in `longva/inference_longva.py`.

For LLaVA-Video-TPO, please following the inference demo in `LLaVA/inference_llava.py`.

### Evaluation

For evaluation, we utilize [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to facilitate a consistent evaluation as previous works. Please refer to their instructions for evaluation setup.

For LongVA-TPO, please refer to `longva/eval.sh` for the evaluation script.

For LLaVA-Video-TPO, please refer to `LLaVA/eval.sh` for the evaluation script.

## Datasets
The TPO dataset for LongVA are available at [Huggingface Dataset](https://huggingface.co/datasets/ruili0/LongVA-TPO-10k).

## Web Demo
To run the web demo for our TPO model (LLaVA-Video-7B-TPO), please run the following python script:
```
conda activate TPOllava
python local_demo/multimodal_chat.py
```


## Training

The training code is coming soon!


## TODO
- [ ] Clean and release the temporal preference data curation pipeline by March.
- [ ] Clean and release the training code by March.

## Citation

If you find this repository useful in your research or work, please consider citing our paper:
```
@misc{li2025temporalpreferenceoptimizationlongform,
      title={Temporal Preference Optimization for Long-Form Video Understanding}, 
      author={Rui Li and Xiaohan Wang and Yuhui Zhang and Zeyu Wang and Serena Yeung-Levy},
      year={2025},
      eprint={2501.13919},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.13919}, 
}
```


## Acknowledgements

This work is based on the original [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA) and [LLaVA-Video](https://github.com/LLaVA-VL/LLaVA-NeXT) repository. We extend our gratitude to the maintainers and contributors of these repositories for their incredible work, which greatly facilitated the development of our project.