import os
import json
import time
from typing import List
import xml.etree.ElementTree as ET
import torch
# import evaluate 
from transformers import pipeline, AutoModelForCausalLM, PreTrainedTokenizerFast
import gc 
# from neo4j import GraphDatabase, RoutingControl 

## NLP metrics
# bleu = evaluate.load('bleu') # completeness 
# rouge = evaluate.load('rouge') # most commonly used
# bertscore = evaluate.load('bertscore') # semantic/correctness
# medcon or use knowledge base to measure conceptual correctness

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

#############################################################################################

def getData(path: str, start=0, stop=None) -> List[str]: 
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
    filename = os.getcwd() + filename
    with open(filename, 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))

def read(filename: str) -> List:
    '''
    filename :: path json file to read and return data from 

    return list of data, read data from filename
    '''
    filename = os.getcwd() + filename
    with open(filename, 'r') as file:
        data = json.load(file)

    return data 

#############################################################################################

def device(model): 
    '''
    check which device (cuda/gpu or cpu) the model is using
    '''
    device = next(model.parameters()).device

    if device.type == 'cuda':
        print("Model using CUDA")
    elif device.type == 'cpu':
        print("Model using CPU")
    else:
        print(f"Model using unknown device type: {device.type}")

def load(model_path: str, template_path: str):
    '''
    model_path :: path to hf config files for local llm 
    template_path :: path to chat template file for tokenizer 

    return model, tokenizer
    ''' 
    # model_path = '/mnt/nfs/CanaryModels/Data/llama-models/models/llama3_1/Meta-Llama-3.1-8B-Instruct'
    # tokenizer_path = '/mnt/nfs/CanaryModels/Data/llama-models/models/llama3_1/Meta-Llama-3.1-8B-Instruct/tokenizer.model'
    
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    # device(model)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    chat_template = open(template_path).read()
    chat_template = chat_template.replace('    ', '').replace('\n', '')
    tokenizer.chat_template = chat_template

    return model, tokenizer

#############################################################################################

def makePrompts(data_path) -> List[List[dict]]:
    '''
    return list of prompts where each prompt (chat history) 
           is structured as a list of dict/json (each dict being a system, user, or assistant message)
    '''
    ehr = "'''".join(getData(data_path))
    # ehr = getData(data_path)

    sysPrompts = ['You are a medical professional', 'You are a neurologist']
    userPrompts = ['Summarize the medical history', "Summarize the medical history related to CAD"]

    prompts = []

    for sp in sysPrompts:
        for up in userPrompts:
            p = [
                {'role': 'system', 'content': sp}, 
                {'role': 'user', 'content': "There are multiple reports which are separated by '''"},
                {'role': 'user', 'content': up}, 
                # {'role': 'user', 'content': ehr[0]}
                {'role': 'user', 'content': ehr}
            ]
            prompts.append(p)
    
    return prompts

def prompt(pipe, messages: List[dict]) -> List[dict]: 
    '''
    messages :: chat history of system, user, and/or assistant messages used to prompt model 

    return model response, completed chat history
    '''
    seq = pipe(
        messages,
        num_return_sequences=1,
        eos_token_id=[128001, 128009],
        max_new_tokens=4096,
        truncation=True,
        do_sample=False,
        repetition_penalty=1.1
    )

    return seq[0]['generated_text']

def gen(model, tokenizer, messages: List[dict]): 
    model_inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors='pt'
    ).to('cuda')
    
    attention_mask = torch.ones_like(model_inputs)
   
    generated_ids = model.generate(
        model_inputs, 
        eos_token_id=[128001, 128009],
        max_new_tokens=4096, 
        do_sample=False,
        attention_mask=attention_mask
    )
    
    model_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return model_outputs

#############################################################################################

def makeTwoStepPrompts(pipe: pipeline, key: str): 
    '''
    pipe :: hf transformer pipeline to use for model inference, Llama3.1
    key :: the key word/condition/term (ex. epilepsy, ex. respiratory) to summarize the patient's history wrt. to
    
    generate and store chat history prompts for two step prompting (adapt prompts from one step prompting stage)
    '''
    summPrompts = read('/prompts/all_records.json')

    keyPrompts = [
        {'role': 'system', 'content' : 'You are a medical professional'},
        {'role': 'user', 'content': 'Give me a comma separated list of the most relevant and most common medical terms, conditions, medications, treatments, and symptoms related to '+key}
    ]

    assistantResponse = prompt(pipe, keyPrompts)[-1]

    chats = []
    for p in summPrompts: 
        chat = keyPrompts + [assistantResponse] + p[1:]
        p[-2]['content'] += ' while retaining any of the terms in the list above only if they existed in the original reports'
        
        chats.append(chat)

    output(chats, '/prompts/two_step_all_records.json')

def inference(pipe: pipeline, chats: List[List[dict]], outfile: str): 
    '''
    pipe :: hf transformers pipeline to use for model inference, Llama3.1
    chats :: list of chat histories/prompts to pass to model and run inference on (chat completion)
    outfile :: name of output file to write model reponses to, filepath relative to current directory 
    '''
    summaries = []
    for chat in chats: 
        summaries.append(prompt(pipe, chat))

    output(summaries, outfile)

def main(): 
    mimiciv_path = '/mnt/nfs/CanaryModels/Data/MIMIC-IV'
    # data_path = '/mnt/nfs/CanarySummarization/Data'
    data_path = '/home/alaina/Sandbox/summarization/data/example_data_one_patient'

    converted_model_path = '/home/alaina/Sandbox/summarization/model/llama'
    chat_template_path = '/home/alaina/Sandbox/summarization/model/chat_template.jinja'

    model, tokenizer = load(converted_model_path, chat_template_path)

    pipe = pipeline(
        'text-generation',
        model=model, 
        tokenizer=tokenizer, 
        torch_dtype=torch.float16, 
        device_map='auto'
    )
    # device(pipe.model)

    start = time.time()

    # one step prompting -> directly asking the model to summarize wrt. key word
    # inference(pipe, read('/prompts/all_records.json'), '/outputs/all_records_prompt.json')

    # two step prompting -> get set of words related to key word, keep set of words when summarizing 
    makeTwoStepPrompts(pipe, 'CAD')
    inference(pipe, read('/prompts/two_step_all_records.json'), '/outputs/two_step/all_records.json')

    print('Execution time (seconds): ', time.time()-start) 

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__": 
    main()
