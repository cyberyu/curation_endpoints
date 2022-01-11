# the following packages to be installed by pip
pip3 install flask
pip install flask-restful
pip install skweak
pip3 install flair
pip install snips_nlu_parsers
pip install pytorch-crf
pip install seqeval
pip3 install hmmlearn
# pip3 install torchcrf
pip3 install pytorch-crf

pip3 install torch~=1.7.1
pip3 install numpy~=1.19.5
pip3 install spacy~=3.0.6
pip3 install transformers~=4.5.1
seqeval~=1.2.2
Flask~=2.0.2
flair~=0.9
ipython~=7.24.1
requests~=2.25.1
pandas~=1.2.5
skweak~=0.2.13


DIR=$PWD
# create a folder under trained_models, and download the pretrained model files there, those files are in gitignore (too large)
cd $DIR/trained_models/distilbert_hmm
wget https://huggingface.co/distilbert-base-uncased/resolve/main/pytorch_model.bin
wget https://huggingface.co/distilbert-base-uncased/resolve/main/config.json

cd $DIR/trained_models/distilroberta_hmm
wget https://huggingface.co/roberta-base/resolve/main/config.json
wget https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin

pip install spacy
# download spacy models
python3 -m spacy download en_core_web_sm
python3 -m spacy download en_core_web_trf
python3 -m spacy download en_core_web_md
