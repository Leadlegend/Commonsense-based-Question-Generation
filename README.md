# Commonsense-based Neural Question Generation Model
This is a seq2seq Question Generation model based on [this](https://github.com/seanie12/neural-question-generation#readme) to implement basic data interface and evaluation. 
However, the model was modified so that it can integrate extern information from Knowledge Graph to assist decoding, and we have got better test results.

## Dependencies
To train or test our model, you should install the following Python Packages:
* python >= 3.7
* pytorch >= 1.5
* nltk(nltk_data files are also needed)
* tqdm
* [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)

## Data Preprocess
Data of Knowledge Graph has been already processed by us, the original KG data is included in `./data/resource.json`

Due to the corpus size, we can not provide SQuAD data on the Github, but you can download the corpus as followed:
```bash
mkdir squad
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O ./data/glove.840B.300d.zip 
unzip ./data/glove.840B.300d.zip 
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O ./squad/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ./squad/dev-v1.1.json
cd data
python process_data.py
```
## Configuration
You might need to change model configuration in ./config.py. <br />
If you want to train with your gpu, please set the gpu device in config.py
Other model configurations and hyper-parameters can also be customized

## Usage
To train the model, you can use the following commandlines:
```bash
python main.py --train (--model_path=<your_model_savepoint_path>)
```
The parameter `--model_path` is optional, if you want to train from scratch, then use `python main.py --train`

Once you model gets the best development set result of current training process, the model parameters will be saved in `./save/train_<timestamp>/<epoch_number>_<dev_loss>`

To test the model, you can use the following commandlines:
```bash
python main.py --model_path=<your_model_paras_path> --output_file=<output_file_path>
```

## Evaluation from this [repository](https://github.com/xinyadu/nqg)
```bash
cd qgevalcap
python2 eval.py --out_file <prediction_file> --src_file <src_file> --tgt_file <target_file>
```

## Currently Best Results
|  <center>BLEU_1</center> |  <center>BLEU_2</center> |  <center>BLEU_3</center> | <center>BLEU_4</center> |
|:--------|:--------:|--------:|--------:|
|<center> 46.30 </center> | <center> 30.85 </center> |<center> 22.76 </center>| <center> 17.63 </center>|

## Reference
[Improving Neural Story Generation by Targeted Common Sense Grounding](https://arxiv.org/abs/1908.09451)
[Commonsense Knowledge Aware Conversation Generation with Graph Attention](https://www.ijcai.org/Proceedings/2018/0643.pdf)
[Knowledge Aware Conversation Generation with Explainable Reasoning over Augmented Graphs](https://www.aclweb.org/anthology/D19-1187)
[Answer-focused and Position-aware Neural Question Generation](https://www.aclweb.org/anthology/D18-1427/)
[Paragraph-level Neural Question Generation with Maxout Pointer and Gated Self-attention Networks](https://www.aclweb.org/anthology/D18-1424)
[Identifying Where to Focus in Reading Comprehension for Neural Question Generation](https://www.aclweb.org/anthology/D17-1219/)


