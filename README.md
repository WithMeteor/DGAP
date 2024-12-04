# PaSIG

The source code of "Dual Graph Adaptive Propagation for Inductive Text Classification".

## File Trees

```
.
├── data
│     ├── embed_bert
│     ├── graph
│     ├── raw
│     │     ├── mr.labels.txt
│     │     ├── mr.texts.txt
│     │     ├── ...
│     │     ├── R52.labels.txt
│     │     └── R52.texts.txt
│     └── temp
├── log
├── out
├── proc
│     ├── dataset_config.py
│     ├── encoder_bert.py
│     ├── graph_builder.py
│     ├── my_tfidf.py
│     └── preprocess_data.py
├── ptm
│     └── bert-base-uncased
│         ├── config.json
│         ├── pytorch_model.bin
│         ├── tokenizer_config.json
│         ├── tokenizer.json
│         └── vocab.txt
└── src
    ├── bert_model.py
    ├── dataset_graph_batch.py
    ├── dataset_graph.py
    ├── dataset_text.py
    ├── gnn_layer.py
    ├── gnn_model_batch.py
    ├── gnn_model.py
    ├── train_bert.py
    ├── train_gnn_batch.py
    ├── train_gnn.py
    └── utils.py
```

## Usage

### 1. Prepare the raw data

Put the raw files in path ```data/raw/```, 
including raw text file (```*.texts.txt```) and label file 
(```*.labels.txt```).

### 2. Process GloVe source file

Download pretrained Glove file from [website](http://nlp.stanford.edu/data/glove.6B.zip).

Unzip the files in `source/` path.

Run the code `proc/handle_glove.py` to process GloVe file.

    `python proc/handle_glove.py`

### 3. Download Stanza source file

[Overview on Github - Stanza](https://github.com/stanfordnlp/stanza)
   
[Source File on Hugging Face - Stanza](https://huggingface.co/stanfordnlp/stanza-en)

Modify the Stanza's path `stanza_path` in `proc/parser.py`, the standalone version will be called in the following way: 
   
   `nlp = stanza.Pipeline(lang='en', processors='tokenize', model_dir=stanza_path, download_method=None)`

### 4. Preprocess data

Run the scripts in the following order (taking dataset MR as an example):

You can choose eight datasets: `mr`, `ohsumed`, `20ng`, `R8`, `R52`, `AGNews`, `dblp`, `TREC`.
   
   + `filter.py` to perform text cleaning;
   
      `python proc/filter.py --dataset mr`
      
      The preprocessed files will be saved at `data/temp/`.
   
   + `parser.py` to perform dependency and constituency analysis on texts;
   
      `python proc/parser.py --dataset mr`
      
      The parse results will be saved at `data/temp/`.
   
   + `constructor.py` to build dual graph of texts.
   
      `python proc/constructor.py --dataset mr`
      
      The graph data will be saved in `data/graph/` path.

### 5. Train DGAP

Run the code ```src/train.py```. For example:
  
  ```shell
  python src/train.py --dataset mr --log_dir log --gpu 0
  ```

The training log of dataset MR will be saved in `log/` path.