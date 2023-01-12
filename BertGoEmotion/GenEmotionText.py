from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
emotions28to4 = {'annoyance':'Angry','anger':'Angry','disgust':'Angry','grief':'Sad',
'sadness':'Sad','disappointment':'Sad','embarrassment':'Sad','remorse':'Sad',
'nervousness':'Sad','fear':'Sad','neutral':'Neutral','confusion':'Neutral','curiosity':'Neutral'
,'surprise':'Neutral','realization':'Neutral','approval':'Neutral','caring':'Neutral','disapproval':'Neutral'
,'admiration':'Neutral','relief':'Happy','gratitude':'Happy','pride':'Happy','optimism':'Happy','desire':'Happy'
,'love':'Happy','joy':'Happy','excitement':'Happy','amusement':'Happy'}
def loadModel(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    classifier = pipeline("text-classification",model=model,tokenizer=tokenizer,device=DEVICE)
    return classifier
def genEmotion(text,model):
    return emotions28to4[model(text)[0]['label']]

if __name__ == '__main__':
    model = loadModel('/media/anlab/DATA/gazotuber/End2End_gazo/BertGoEmotion/modelBert')
    print(genEmotion('angry',model))
