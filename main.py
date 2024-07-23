import os, json
import xml.etree.ElementTree as ET
import evaluate #from hugging face 
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage
from llama_index.llms.mistralai import MistralAI
from llama_index.core.llms import ChatMessage
from openai import OpenAI

# data_path = '/mnt/nfs/CanarySummarization/Data'
data_path = 'C:/Users/alain/Research/MedDB/data/training-PHI-Gold-Set1'   

## models
# api key? mistral account? 
# model = ''
# client = MistralClient()

# llm = MistralAI(model=model)

## NLP metrics
bleu = evaluate.load('bleu') # completeness 
rouge = evaluate.load('rouge') # most commonly used
bertscore = evaluate.load('bertscore') # semantic/correctness
# medcon or use knowledge base to measure conceptual correctness

def getData(path: str) -> list[str]: 
    '''
    path :: directory path to the dataset

    return list of medical reports in the path folder 
    '''
    reports = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
            xml_text = file.read()
            root = ET.fromstring(xml_text)
            text_content = root.find('.//TEXT').text
            reports.append(text_content)      
    return reports

def update(data, file):
    '''
    update the dict stored in file, maps {prompt: summary}
    '''
    with open(file, 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))

client = OpenAI()

def prompt(sysPrompt: str, userPrompt: str) -> str:
    '''
    sysPrompt :: desired behavior/role of the model, domain specification, ex. 'doctor'  
    userPrompt :: instructions to model, ex. 'summarize the following documents'

    return model response to prompts (summary)
    '''
    # denote assistant messages for in-context learning? or wrap in userPrompt
    # adjust temperature to be lower? default 1
    response = client.chat.completions.create(
                    model='gpt-4-turbo', 
                    messages=[
                        {'role': 'system', 'content': sysPrompt},
                        {'role': 'user', 'content': userPrompt}
                    ]
                ).choices[0].message.content
    return response

def main(): 
    ehr = "'''".join(getData(data_path))
    sysPrompts = ['You are a medical professional', 'You are a neurologist', 'You are a medical professional specializing in neurology']
    userPrompts = ['Summarize the medical history', 'Summarize the neurological medical history', "Summarize the patient's medical history related to CAD"]

    for sp in sysPrompts: 
        for up in userPrompts: 
            summary = prompt(sp, up + ehr)
            info = {
                'systemPrompt': sp,
                'userPrompt': up,
                'summary': summary
            }
            update(info, 'output.json')

main()

