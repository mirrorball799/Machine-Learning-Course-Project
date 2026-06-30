# 家庭电力消耗多变量时间序列预测

2026 年专硕《机器学习》课程考核项目。

## 任务

基于 UCI Individual Household Electric Power Consumption 数据集，对家庭总有功功率进行多变量时间序列预测：

- **输入**：过去 90 天的多变量序列（19 维特征）
- **输出**：未来 90 天（短期）或 365 天（长期）的有功功率曲线
- **模型**：LSTM、Transformer、SCA-Net（改进模型）

## 项目结构

```
course_project/
├── main.py                     # 主入口
├── run_ablation.py             # 消融实验入口
├── requirements.txt
│
├── data/                       # 数据文件
│   └── weather_monthly.csv     # 月度天气数据 (Météo-France)
│
├── data_processing/
│   ├── preprocess.py           # 数据加载/清洗/日聚合/标准化
│   ├── dataset.py              # PyTorch Dataset + DataLoader
│   └── weather_download.py     # 天气数据下载工具
│
├── models/
│   ├── lstm_model.py           # LSTM Seq2Seq
│   ├── transformer_model.py    # Transformer Encoder-Decoder
│   ├── improved_model.py       # SCA-Net (改进模型)
│   └── ablation_models.py      # 消融实验变体
│
├── training/
│   └── trainer.py              # 训练循环 + 多GPU + 早停
│
├── utils/
│   ├── config.py               # 全局配置
│   └── visualization.py        # 绘图工具
│
└── scripts/
    ├── run_lstm.sh             # LSTM 实验
    ├── run_transformer.sh      # Transformer 实验
    ├── run_sca.sh              # SCA-Net 实验
    ├── run_all.sh              # 全部模型
    └── run_ablation.sh         # 消融实验
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 快速验证 (dry-run, 仅检查前向传播)
python main.py --model lstm --task short --dry-run

# 单模型实验
python main.py --model lstm --task both

# 全部模型
bash scripts/run_all.sh both
```


## 数据

- 原始电力数据需从 [UCI Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) 下载，放置于项目父目录
- 天气数据已包含在 `data/weather_monthly.csv`，来源 Météo-France


## 环境

| 项目 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | 2.9+ (CUDA 12.8) |
| pandas | 2.x |
| numpy | 1.x |
| matplotlib | 3.x |
