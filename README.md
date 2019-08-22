## This repo contains development code for CANDLE Milestone 14.

There two main scripts you need to run:
(1) Script that genrates topN dataset and the data splits (build_topN.py)
(2) Script that train NN model (main_m14.py)

Script (1) requires to have a folder called "data" and some required files.
You can just copy the folder /vol/ml/apartin/projects/candle/data to your parent dir.

First, run script (1) as follows:
```py
python build_topN.py --top_n 21 --format parquet --labels
```
This will create dir called top21_data

Then, run script (2) to train with SGD:
```py
python main_m14.py --dirpath top21_data --opt sgd
```
This will create dir called top21_out

Or, run script (2) to train with CLR:
```py
python main_m14.py --dirpath top21_data --clr_mode trng1
```

