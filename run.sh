pip install -r requirements.txt

cp -r /home2/pytorch-broad-models/gpuoob/FastSpeech2/ckpt output/.
cp /home2/pytorch-broad-models/gpuoob/FastSpeech2/generator_LJSpeech.pth.tar hifigan

python3 synthesize.py --source preprocessed_data/LJSpeech/val.txt --restore_step 900000 --mode batch -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml --device cuda --precision float16 --jit --nv_fuser --profile
