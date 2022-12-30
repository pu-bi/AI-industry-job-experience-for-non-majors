from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
import torch
import torch.nn.functional as F
import numpy as np
import random

# Load the model and tokenizer
def load_model(model_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(model=model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return trainer, tokenizer

# Make reply
def make_sentence(intention):
    id2label = {0: '활동', 1: '숙소', 2: '맛집', 3: '날씨', 4: '카페', 5: '인사', 6: '관광지추천', 7: '소개'}
    if intention not in id2label.values(): # threshold 미만이므로 모르겠다는 것
        sentence = "다시 한번 질문해 주십시오. 답변 가능한 질문은 '활동', '숙소', '맛집', '날씨', '카페', '인사', '관광지추천', '소개' 입니다."
        return sentence

    if intention == "활동":
        sentence = f"{intention}이(가) 궁금하시군요!"

    elif intention == "숙소":
        sentence = f"{intention}이(가) 궁금하시군요!"

    elif intention == "맛집":
        sentence = f"{intention}이(가) 궁금하시군요!"

    elif intention == "날씨":
        sentence = f"{intention}이(가) 궁금하시군요!"

    elif intention == "카페":
        sentence = f"{intention}이(가) 궁금하시군요!"

    elif intention == "인사":
        sentence = f"{intention}이(가) 궁금하시군요!"

    elif intention == "관광지추천":
        weather = ['봄', '여름', '가을', '겨울']
        choice_weather = random.choice(weather)
        attractions = ['아르떼 뮤지엄', '사려니숲길', '어승생악', '수산저수지', '원 앤 온리', '카멜리아 힐', '수월봉']
        choice_attractions = random.choice(attractions)
        sentence = f'이번 {choice_weather}, {choice_attractions} 가 보시는 건 어때요?'

    elif intention == "소개":
        sentence = "소개"

    else:
        sentence = "오류가 발생했습니다"
    return sentence

# Make the answer
def answer_system(trainer, tokenizer, question: str) -> str:
    id2label = {0: '활동', 1: '숙소', 2: '맛집', 3: '날씨', 4: '카페', 5: '인사', 6: '관광지추천', 7: '소개'}
    encoding = tokenizer(question, return_tensors="pt")
    encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}
    outputs = trainer.model(**encoding)
    logits = outputs.logits
    probs = logits.squeeze().cpu()
    labels_predict = [ i if i > 0 else 0 for i in probs ]

    softmax_probs = F.softmax(probs, dim=-1)

    labels_predict = np.where(softmax_probs==softmax_probs.max())[0][0]
    pred = id2label[labels_predict] # 질문에 모델의 Intention

    if softmax_probs[labels_predict] < 0.5:
      pred = 'unpredictable'

    sentence = make_sentence(pred)
    return sentence
