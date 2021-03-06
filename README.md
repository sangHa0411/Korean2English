# Korean - English Translation

## Modules
  ```
  |-- Log
  |-- Model
  |-- Token
  |   |-- en_tokenizer.model
  |   |-- en_tokenizer.vocab
  |   |-- english.txt
  |   |-- kor_tokenizer.model
  |   |-- kor_tokenizer.vocab
  |   `-- korean.txt
  |-- collator.py
  |-- dataset.py
  |-- model.py
  |-- preprocessor.py
  `-- train.py
  ```

## Tokenizer Speicication
  1. Subword Tokenization
  2. SentencePiece
  3. Vocab size
      * English : 7000
      * Korean : 7000
  
## Model Specification
  1. Structure : Transformer 
  2. Layer_size : 6
  3. Embedding size : 512
  4. Hidden size : 2048
  5. Seq size : 30
  6. Head size : 8

## Training
  1. epoch_size : 100
  2. batch_size : 64
  3. validation_ratio : 0.2
  4. warmup_steps : 4000

## Source
  1. aihub : https://aihub.or.kr/aihub-data/natural-language/about
  2. 한국어 - 영어 변역 말뭉치
  
