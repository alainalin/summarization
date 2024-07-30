import os
import json
import xml.etree.ElementTree as ET
import torch
import evaluate 
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizerFast, PreTrainedTokenizerFast
from transformers import StoppingCriteriaList, EosTokenCriteria

# data_path = '/mnt/nfs/CanarySummarization/Data'
data_path = '/home/alaina/Documents/summarization/example_data_one_patient'

# model_path = '/mnt/nfs/CanaryModels/Data/llama-models/models/llama3_1/Meta-Llama-3.1-8B-Instruct'
# tokenizer_path = '/mnt/nfs/CanaryModels/Data/llama-models/models/llama3_1/Meta-Llama-3.1-8B-Instruct/tokenizer.model'
converted_model_path = '/home/alaina/Documents/summarization/llama'

model = AutoModelForCausalLM.from_pretrained(converted_model_path, device_map='auto',)
tokenizer = PreTrainedTokenizerFast.from_pretrained(converted_model_path)

#############################################################################################

## NLP metrics
bleu = evaluate.load('bleu') # completeness 
rouge = evaluate.load('rouge') # most commonly used
bertscore = evaluate.load('bertscore') # semantic/correctness
# medcon or use knowledge base to measure conceptual correctness

#############################################################################################

def getData(path: str, start=0, stop=None) -> list[str]: 
    '''
    path :: directory path to the dataset to extract from

    start :: int, start getting reports at this index (inclusive, 0 index)
    stop :: int, stop getting reports at this index (exclusive, 0 index)

    return data from [start, stop) ~indices
    '''
    reports = []
    listdir = os.listdir(path)

    if stop is None: 
        stop = len(listdir)

    if stop < start or start >= len(listdir) or stop <= 0 or stop > len(listdir): 
        raise ValueError('Invalid bounds to fetch data') 

    for i in range(start, stop):
        with open(os.path.join(path, listdir[i]), 'r', encoding='utf-8') as file:
            xml_text = file.read()
            root = ET.fromstring(xml_text)
            text_content = root.find(".//TEXT").text
            reports.append(text_content)
            
    return reports

def output(data, filename: str):
    '''
    data :: data to convert into and store in json file
    filename :: name of file to write to (create new file or rewrite existing if name already in use)
    
    return nothing, output data into filename 
    '''
    with open(filename, 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))

def read(filename: str) -> list:
    '''
    filename :: path json file to read and return data from 

    return list of data, read data from filename
    '''
    with open(filename, 'r') as file:
        data = json.load(file)

    return data 

#############################################################################################

def makePrompts() -> list[list[dict]]:
    '''
    return list of prompts where each prompt (chat history) 
           is structured as a list of dict/json (each dict being a system, user, or assistant message)
    '''
    # ehr = "'''".join(getData(data_path))
    ehr = getData(data_path)

    sysPrompts = ['You are a medical professional', 'You are a neurologist']
    userPrompts = ['Summarize the medical history: ', "Summarize the medical history related to CAD: "]

    prompts = []

    for sp in sysPrompts:
        for up in userPrompts:
            p = [
                {'role': 'system', 'content': sp}, 
                {'role': 'user', 'content': up}, 
                {'role': 'user', 'content': ehr[0]}
            ]
            prompts.append(p)
    
    return prompts

def prompt(pipe, messages: list[dict]) -> list[dict]: 
    '''
    messages :: chat history of system, user, and/or assistant messages used to prompt model 

    return model response, completed chat history
    '''
    # tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True)

    seq = pipe(
        messages,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=500,
        truncation=True,
        temperature=0.1, 
        # do_sample=False,
        add_special_tokens=False, 
        repetition_penalty=1.1, 
    )

    return seq[0]['generated_text']

def main(): 
    pipe = pipeline(
        'text-generation', # summarization pipeline not supported for llama
        model=model, 
        tokenizer=tokenizer, 
        torch_dtype=torch.float16, 
        device_map='auto',
    )

    prompts = makePrompts()

    summaries = []
    for p in prompts: 
        summaries.append(prompt(pipe, p))

    output(summaries, 'output_one_record_gen.json')

main()