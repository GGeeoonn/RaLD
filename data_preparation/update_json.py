import json
import spacy
from tqdm import tqdm
import pyinflect

# spaCy에 inflect 확장 등록 (force=True 추가)
spacy.tokens.Token.set_extension('inflect', getter=lambda token: token._.inflect, force=True)

# 전역 변수로 한 번만 로드
nlp = spacy.load("en_core_web_sm")

def extract_verbs(caption):
    doc = nlp(caption)
    verbs = []

    for token in doc:
        if token.pos_ == "VERB":
            verb = token.text
            if not verb.endswith('ing'):
                verb = verb + 'ing'
            verbs.append(verb)
    
    return list(dict.fromkeys(verbs))

def update_json_with_verbs():
    input_path = "/root/MIGC/data_preparation/result.json"
    output_path = "/root/MIGC/data_preparation/result_verb.json"

    with open(input_path, 'r') as f:
        data = json.load(f)

    print("Data structure:", type(data))
    first_key = list(data.keys())[0]
    print("Sample data structure for first key:", type(data[first_key]))
    print("First item content:", data[first_key][0])

    for key in tqdm(data.keys()):
        for idx, item in enumerate(data[key]):
            caption = item.get('caption', '') or item.get('origin_caption', '')
            if not caption:
                print(f"Available keys in item: {item.keys()}")
                print(f"Warning: No caption found for key {key}, index {idx}")
                continue

            verbs = extract_verbs(caption)
            data[key][idx]['verbs'] = verbs

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Updated JSON saved to {output_path}")

    # 예시 출력
    first_key = list(data.keys())[0]
    print("\nSample Result:")
    print(f"Caption: {data[first_key][0]['caption']}")
    print(f"Verbs: {data[first_key][0]['verbs']}")

if __name__ == "__main__":
    update_json_with_verbs()