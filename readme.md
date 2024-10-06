
---

<div align="center">
 <img src="img/logo.png" width="400px">
</div>

---

# Prediction System Based on SoC Static Voltage Drop Big Data

SoC Analyser is a high-performance analysis system for SoC static voltage drop big data. It accurately predicts voltage drops at various chip locations based on a chip feature map, which is significant for large-scale integrated circuit design.

<div align="center">
 <img src="img/process.png" width="800px">
</div>

SoC Analyser achieves an average error of 0.0008 and a 92% correlation coefficient in static voltage drop prediction tasks for five mainstream chip designs, running faster than static analysis tools.

You can reproduce our experiments with the following steps.

---

## :sunny: Environment Setup

1. Enter the project directory
```shell
git clone https://github.com/HeartyHaven/soc-Analyzer
cd soc-Analyzer
```

2. Install dependencies: Create an environment
```shell
conda create -n soc_analyser python==3.9.16
conda activate soc_analyser
pip install --upgrade pip # enable PEP 660 support
```

3. Install dependencies
```shell
pip install -r requirements.txt
```

## :rocket: Data Acquisition

We use the GloryBolt EMIR analysis tool to perform static voltage drop analysis on extracted results of integrated circuit netlists. A dataset with 7322 netlist extraction results is provided, which you can download from [Baidu Netdisk](https://pan.baidu.com/s/1Uz7mPTMExlROH5i5W4sN2Q?pwd=8xa8): Download all tar.gz files and save them in the same folder named `dataset`.

After downloading, execute
```shell
cd dataset
for file in *.tar.gz; do tar -xzvf "$file"; done
```

## :hourglass: Data Preprocessing

#### Input Data

After downloading, store the input data in the following format:
```yaml
-data
| - nvdla-small_freq_200_mp_1_
|  |- eff_res.rpt.gz
|  |- min_path_res.rpt.gz
|  |- power.rpt.gz
| - nvdla-small_freq_200_mp_2_
| - nvdla-small_freq_200_mp_3_
 ...
```
Then execute the following command
```shell
python feature_extraction/process_data.py \
--data_root path/to/your/data
```

This extracts key information from your data and saves it locally in array form.

#### Dataset Generation

```shell
python feature_extraction/generate_training_set.py 
```

#### Generate Index

```shell
python feature_extraction/generate_csv.py 
```
This generates index files `train.csv` and `test.csv`.

## :star: Model Training

Pre-trained model parameters `model.pth` are provided and can be used directly for testing.

**Training from Scratch:** Use `batch_size`=1170 to train the SoC Analyser model for about 12 hours on 8*RTX 4090 devices. We observed that `batch_size` is positively correlated with training effectiveness, so reducing `batch_size` may degrade performance.

Before training, modify the parameters `save_path`, `ann_file`, `data_root` in `args/train.json` to specify the result save path, `train.csv` path, and the path where the extracted feature files are located.

To initialize the model with pre-trained parameters, define `pretrained`.
```shell
python model_training/train.py --args model_training/args/train.json
```

## :black_nib: Model Prediction and Evaluation

Before prediction, modify the parameters `save_path`, `ann_file`, `data_root` in `model_training/args/test.json` to specify the result save path, `test.csv` path, and the path where the extracted feature files are located.
```shell
python model_training/test.py --args model_training/args/test.json --pretrained path/to/your/model.pth
```
The program outputs detailed model evaluation information, including the mean absolute error and correlation coefficient for different designs. You will get the voltage drop distribution of samples at the `save_path`.

## :moon: Acknowledgments

The implementation of this project references the excellent work from the [CircuitNet](https://github.com/circuitnet/CircuitNet) and [IREDGe](https://github.com/VidyaChhabria/ThermEDGe-and-IREDGe.git) projects. We express our sincere gratitude for their outstanding contributions!
