python -m pip install -U torch torchvision torchaudio

python -m pip install -U transformers datasets peft accelerate tqdm sentencepiece safetensors

have a training_data.jsonl where each json line has a input and output section like

{"input": "GoodBye", "output": "Bye"}

{"input": "do this question", "output": "ok doing"}

run train.py and after its done do chat.py

the more lines you have in training data, the more the bot talks like you, give atleast 250k lines for good results
