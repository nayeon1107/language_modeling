# Language Modeling

## 📚 파일 구조
```bash
📂 generated
 ┃ ┗ 📜 shakespeare_by_LSTM_T=0.1.txt
 ┃ ┗ 📜 shakespeare_by_LSTM_T=0.3.txt
 ┃ ┗ 📜 ...
 ┃ ┗ 📜 shakespeare_by_RNN_T=10.txt
📂 models
 ┃ ┗ 📜 best_state_LSTM_epoch97.pth
 ┃ ┗ 📜 best_state_RNN_epoch99.pth
 ┣ 📜 dataset.py
 ┣ 📜 generate.py
 ┣ 📜 main.py
 ┗ 📜 model.py
```

## 📃 Data
 **Shakespeare Dataset**



```bash
# shakespeare_train.txt

First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
```

---

## 🔗 Models 
- Vanila RNN
- LSTM

### Parameters
- embedding_dim = 128
- hidden_dim = 128
- n_layers = 1
&nbsp; &nbsp;


## 📊 Compare Result
### 🔍 Compare RNN & LSTM
![trainvalidloss](https://github.com/nayeon1107/language_modeling/assets/88521667/e0045051-30af-48c8-b3a3-3a487a43e23a)
```bash
▶ Train, Valid 모두 LSTM 이 Vanila RNN 보다 낮은 loss를 보임
```
&nbsp; &nbsp;
### 🔍 Compare Softmax Temperature T

---

패키지 다운로드
```python
pip install requirements.txt
```

모델 실행
```python
python main.py {model_to_use} # RNN, LSTM
```
