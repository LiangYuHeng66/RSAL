# Reliable Semantic Alignment Learning for Text-based Aerial-Ground Person Retrieval


<div align="center">

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#news">News</a> •
  <a href="#requirements">Requirements</a> •
  <a href="#dataset-preparation">Dataset</a> •
  <a href="#training-and-evaluation">Training & Evaluation</a> •
  <a href="#citation">Citation</a> •
  <a href="#acknowledgments">Acknowledgments</a> •
  <a href="#license">License</a>
</p>

</div>

---

<a id="introduction"></a>
## 📌 Introduction
This repository provides the official implementation of Reliable Semantic Alignment Learning for Text-based Aerial-Ground Person Retrieval.

<a id="news"></a>
## 📢 News
- **[Coming Soon]** Core implementation and pretrained weights will be released.

<a id="requirements"></a>
## 🛠️ Requirements
We train and evaluate RSAL on a single NVIDIA RTX 4090 GPU with 24GB memory. The main dependencies are:

```bash
pytorch 1.9.0
torchvision 0.10.0
prettytable
easydict
```


<a id="dataset-preparation"></a>
## 📂 Dataset Preparation
We conduct experiments on multiple text-based person retrieval benchmarks:
- **TAG-PEDES** for Text-based Aerial-Ground Person Retrieval.
- **AERI-PEDES** and **TBAPR** for Text-based Aerial Person Retrieval.
- **CUHK-PEDES**, **ICFG-PEDES**, and **RSTPReid** for conventional Text-based Person Retrieval.

Please obtain the datasets from their official repositories:
- **TAG-PEDES**: [official repository](https://github.com/Flame-Chasers/TAG-PR/tree/main)
- **TBAPR / AERI-PEDES**: [official repository](https://github.com/xbdxwyh/AEA-FIRM-main)
- **CUHK-PEDES**: [project page](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)
- **ICFG-PEDES**: [official repository](https://github.com/zifyloo/SSAN)
- **RSTPReid**: [official repository](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Please organize the datasets as follows:
```text
dataset/
├── TAG-PEDES/
│   ├── anno_dir/
│   │   ├── train_reid.json
│   │   └── test_reid.json
│   └── images/
│       ├── 0001.jpg
│       ├── 0002.jpg
│       └── ...
├── AERI-PEDES/
├── TBAPR/
├── CUHK-PEDES/
├── ICFG-PEDES/
└── RSTPReid/
```

<a id="training-and-evaluation"></a>
## 🚀 Training and Evaluation

### Training New Models

To train RSAL, simply run:

```bash
sh run_rsal.sh
```

### Evaluation
```bash
python test.py --config_file 'path/to/model_dir/configs.yaml'
```

<a id="citation"></a>
## 📝 Citation
If you find this code useful for your research, please cite our paper.

```bibtex
Manuscript under review
```

<a id="license"></a>
## 📄 License
This project is released under the MIT License.


<a id="acknowledgments"></a>
## 🙏 Acknowledgments
Some components of this code implementation are adopted from [IRRA](https://github.com/anosorae/IRRA). We sincerely appreciate for their contributions.





