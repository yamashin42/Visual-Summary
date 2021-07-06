# Visual-Summary
Sample code for Visual Summary Identification from Scientific Publications via Self-Supervised Learning.
As we do not retain the right to distribute the data used for training, we release the code with sample training data.

## Dependency
* pytorh 1.7.1
* spacy 2.3.2
* transformers 2.11.0
* tensorboardX 2.0

## Training Data Preparation
We placed the sample training data in "data/train". For each figure, single JSON file including figure caption and paragraph which mentions figure is created like "data/train/1/Figure_1.json". Each JSON file should be named "Figure_\[number\].JSON. Figures from the same paper should be placed in the same directory. For example, "data/train/1" contains all figures from a single paper.

## Training
As we do not retain the right to distribute all samples for training, we only provide the sample instances in "data/train".
### PubMed
We collected the original papers of the data available here https://github.com/viziometrics/centraul_figure
### CS
We collected publicly availabel papers though we do not retain the right for some of them.<br>
Example<br>
* CVF: https://openaccess.thecvf.com/menu
* ACLAnthology: https://aclanthology.org/ <br>
There are also several conferences providing proceedings as open access.
### Paper parsing
We need a figure caption and a paragraph which mentions figure for model training.
To obtain a figure caption, we used DeepFigures library available here (https://github.com/allenai/deepfigures-open).
To obtain a paragraph which mentions figure, we used ScienceParse library available here (https://github.com/allenai/science-parse).

## Test Data Preparation
We prepared the sample test data in "data/test/test_sample.json". The JSON file needs to contain abstract and captions of all figures from a single paper.

## Evaluation Data
### PubMed
We used the data available here https://github.com/viziometrics/centraul_figure
### CS
Please send an e-mail to s.yamamoto(at)fuji.waseda.jp

## Inference
python test.py --test_data /path/to/test/data.json --model /path/to/model/weight --bert /path/to/BERT/model

## Trained Model
We used scibert_base_uncased from here: https://github.com/allenai/scibert
Under Preparation
