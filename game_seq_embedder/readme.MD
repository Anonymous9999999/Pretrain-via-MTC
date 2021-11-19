## Install From Git
pip install git+ssh://git@gitlab.leihuo.netease.com:32200/pujiashu/game_seq.git

## Install From local
python setup.py install
python3.6 setup.py install

## Install relevant python packages
You may refer to requirements.txt, however, you DO NOT need to install transformers packages yourself.

## Load model from shuyuan
For shuyuan object storage SDK doc, you can refer to https://confluence.leihuo.netease.com/pages/viewpage.action?pageId=36776629

```python
# Find all avaliable models in shuyuan
prefix = 'game_seq'
avaliable_models_urls = query(prefix) # query function from shuyuan SDK
print(avaliable_models_urls) # each model is compressed as a tar.gz file

# You just have to download models from shuyuan and unarchive them as the model dirs
```

## Quick Start
```python
from game_seq_embedder import init_behavior_sequence_embedder

# set model dir
# In model dir, you can found meta_config.json, which include model type, tokenizer type, and other meta information.
# For model information, you can refer to config.json
model_dir = 'game_bert_time_embed_whitespace'

# load sample (This is an example)
# Caution! The length of each sample must be an integer multiple of 3. 
# In raw sequences, the design id may be shown as '/' or not shown, in such cases, it should be replaced by '0'.
# Each sequence should represent gameid_0, designid_0, timestamp_0, ..., gameid_i, designid_i, timestamp_i
batch_sample = [['400347', '0', '1604578001', '400616', '0', '1604578002'],
['400347', '0', '1604578003', '400347', '0', '1604578004'],
['400000', '0', '1604578037', '400121', '0', '1604578039']]

# init embedder
embedder = init_behavior_sequence_embedder(model_dir)

# get embeddings for sample
# batch_size: Maximum number of samples loaded into GPU at a time
# layer: the position of layer where mean pooling is applied
embedding = embedder.embed(batch_sample,
                           batch_size = 4,
                           layer= -2) # output shape: batch_size x 768

# ----------------------------------------------------------------------------------
# For finetuning this model, just set is_finetune to true
# ----------------------------------------------------------------------------------
embedder = init_behavior_sequence_embedder(model_dir, is_finetune=True)

# training code simple example ...
from torch import optim
# Here you could add the parameters' of the downstream model to the optimizer as well
optimizer = optim.SGD(embedder.model_params, lr=0.01, momentum=0.9)
optimizer.zero_grad()
# ... some loss function here
# loss.backward()
optimizer.step()

# Don't forget to set to feature extraction mode after training, and set the mean pooling layer to -1
embedder.set_to_feature_extration_mode()
embeddings = embedder.embed(batch_sample, layer=-1)

```