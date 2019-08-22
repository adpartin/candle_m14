## This repo contains development code for CANDLE Milestone 14.

There are two main scripts that you need to run:

1) Script that generates topN dataset and the data splits (build_topN.py)
2) Script that trains NN model (main_m14.py)

Script (1) requires to have a folder called ./data with certain files.
You can just copy the folder /vol/ml/apartin/projects/candle/data to the main dir.

First, run script (1) as follows:
```
python build_topN.py --top_n 21 --format parquet --labels
```
This will create a dir `./top21_data`

Then, run script (2) to train with SGD:
```
python main_m14.py --dirpath top21_data --opt sgd
```
This will create dir `./top21_out`. Inside `top21_out`, you'll find a dir for this specific run.

Or, run script (2) to train with CLR:
```
python main_m14.py --dirpath top21_data --clr_mode trng1
```

