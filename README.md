## Setup
```bash
pip install -r requirements.txt
```

## Usage

### 1. Interactive Menu (Recommended)
```bash
python run.py
```
Follow the numbered menu steps:
1. Preprocess data
2. Create partitions  
3. Train centralized model
4. Run federated training
5. View results

### 2. Manual Commands

**Preprocess data:**
```bash
python -m app.main preprocess --data-path "Data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv,Data/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv,Data/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
```

**Create partitions:**
```bash
python -m app.main partition --clients 10
```

**Train centralized model:**
```bash
python -m app.main central-train --epochs 5
```

**Run federated learning:**
```bash
python -m app.main fl-server --clients 10 --rounds 5
```

## Results
- Training outputs saved in `runs/` folder
- Metrics plots and models included
- Compare centralized vs FL performance

## Requirements
- Python 3.11+
- Docker for federated training
- CICIDS2017 dataset in "Data/" folder
