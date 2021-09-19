# Korean - English Translation
  1. Driectory Structure
  ```
  |-- Log
  |   `-- events.out.tfevents.1632020811.abf6efe47668.31225.0
  |-- Model
  |   `-- checkpoint_transformer.pt
  |-- Token
  |   |-- en_tokenizer.model
  |   |-- en_tokenizer.vocab
  |   |-- english.txt
  |   |-- kor_tokenizer.model
  |   |-- kor_tokenizer.vocab
  |   `-- korean.txt
  |-- __pycache__
  |   |-- collator.cpython-38.pyc
  |   |-- dataset.cpython-38.pyc
  |   |-- model.cpython-38.pyc
  |   `-- preprocessor.cpython-38.pyc
  |-- collator.py
  |-- dataset.py
  |-- model.py
  |-- preprocessor.py
  `-- train.py
  ```
  2. Soruce : Korean, Target : English

## Embedding
  1. Subword Tokenization
  2. SentencePiece
  3. Wor2Vec - SkipGram

## Model Specification
  1. layer_size : 6
  2. d_model : 512
  3. hidden_size : 2048
  4. max_size : 30
  5. head_size : 8

## Train Speicication
  1. epoch_size : 100
  2. batch_size : 64
  3. validation_ratio : 0.2
  4. warmup_steps : 4000


