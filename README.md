# Reliable Semantic Alignment Learning for Text-based Aerial-Ground Person Retrieval

Official PyTorch implementation of the paper Reliable Semantic Alignment Learning for Text-based Aerial-Ground Person Retrieval.

<div align="center">

<p align="center">
  <a href="#-news">News</a> •
  <a href="#-introduction">Introduction</a> •
  <a href="#-setup">Setup</a> •
  <a href="#-dataset-preparation">Dataset</a> •
  <a href="#-configuration">Config</a> •
  <a href="#-training">Training</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

---

## 📢 News


# Usage

## Requirements
we use a single RTX4090 24G GPU for training and evaluation. 
```
pytorch 1.9.0
torchvision 0.10.0
prettytable
easydict
```

### Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json

|-- your dataset root dir/
├── TAG-PEDES/
│   ├── images/
│   ├── train.json
│   ├── test.json
│   └── ...
├── AERI-PEDES/
├── TBAPR/
├── CUHK-PEDES/
├── ICFG-PEDES/
└── RSTPReid/


```



## Acknowledgments
Some components of this code implementation are adopted from [IRRA](https://github.com/anosorae/IRRA), [MLLM4Text-ReID](https://github.com/WentaoTan/MLLM4Text-ReID), and [HAM](https://github.com/sssaury/HAM). We sincerely appreciate for their contributions.
