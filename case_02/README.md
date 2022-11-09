# Company names matcher
### Models:
*   CatBoost
*   Bert
*   SentenceBert

### Quick start
```
# clone repository
git clone https://github.com/Ruslan515/ITMO_DL_practice

cd ITMO_DL_practice/case_02

# install dependencies
pip install -r requirements.txt

# run pipeline
python main.py
```
*   Use [similar_companies.ipynb](https://github.com/Ruslan515/ITMO_DL_practice/blob/main/case_02/similar_companies.ipynb) to demonstrate results

### Metrics
Models   |  F1 Macro
  ---    |    ---              
CatBoost |   0.85 |
Bert     |   0.98 |

### Perfomance
#### SentenceBert:
GPU: Tesla T4
Inference speed: 4500 sentences of length 128 per second
