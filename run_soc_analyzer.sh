git clone https://github.com/HeartyHaven/soc-Analyzer
cd soc-Analyzer
conda create -n soc_analyser python==3.9.16
conda activate soc_analyser
pip install --upgrade pip # enable PEP 660 support
pip install -r requirements.txt
cd ../dataset
for file in *.tar.gz; do tar -xzvf "$file"; done #unzip all tar files
cd ../soc-Analyzer
python feature_extraction/process_data.py --data_root ../dataset
python feature_extraction/generate_training_set.py 
python feature_extraction/generate_csv.py 
python model_training/train.py --args model_training/args/train.json
python test.py --args model_training/args/test.json --pretrained ./model.pth