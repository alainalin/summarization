import os, json
import xml.etree.ElementTree as ET
import torch
import evaluate 
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer

# data_path = '/mnt/nfs/CanarySummarization/Data'
data_path = '/home/alaina/Documents/summarization/example_data_one_patient'

# model_path = '/mnt/nfs/CanaryModels/Data/llama-models/models/llama3_1/Meta-Llama-3.1-8B-Instruct'
# tokenizer_path = '/mnt/nfs/CanaryModels/Data/llama-models/models/llama3_1/Meta-Llama-3.1-8B-Instruct/tokenizer.model'
converted_model_path = '/home/alaina/Documents/summarization/llama'

model = LlamaForCausalLM.from_pretrained(converted_model_path, device_map='auto',)
tokenizer = LlamaTokenizer.from_pretrained(converted_model_path)

def testGen(): 
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
    model_inputs = tokenizer(['Summarize the following advice: You can make a simple Caprese salad with fresh tomatoes, basil, and mozzarella cheese. Alternatively, you can make a bruschetta by toasting some bread, topping it with diced tomatoes, basil, and mozzarella cheese, and drizzling with olive oil'], return_tensors="pt", padding=True)

    generated_ids = model.generate(**model_inputs, max_new_tokens=20)
    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

def testPipe(): 
    pipe = pipeline(
        'text-generation', # summarization is not supported for LlamaForCasualLM
        model=model, 
        tokenizer=tokenizer, 
        torch_dtype=torch.float16, 
        device_map='auto',
    )

    chat = [
        {'role': 'system', 'content': 'You answer with the most minimal and essential information, providing concise and coherent response without extra word fluff.'}, 
        {'role': 'user', 'content': 'Summarize the following advice'}, 
        {'role': 'user', 'content' : 'You can make a simple Caprese salad with fresh tomatoes, basil, and mozzarella cheese. Alternatively, you can make a bruschetta by toasting some bread, topping it with diced tomatoes, basil, and mozzarella cheese, and drizzling with olive oil'}
    ]

    tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True)

    sequences = pipe(
        tokenized_chat,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=50,
        truncation=True,
        temperature=0.1, 
        do_sample=True
    )
    for seq in sequences:
        print(f"{seq['generated_text']}")

testGen()
testPipe()
# text-generation task --> not summarizing even with explicit prompting 
# run inference locally with script? llama-cpp-python requires .gguf file (other documentation depreciated)

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
    output data into filename (create new file or rewrite existing if name already in use)
    '''
    with open(filename, 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))

def prompt(sysPrompt: str, userPrompt: str) -> str:
    '''
    sysPrompt :: desired behavior/role of the model, domain specification, ex. 'doctor'  
    userPrompt :: instructions to model, ex. 'summarize the following documents'

    return model response to prompts (summary)
    '''
    # denote assistant messages for in-context learning? or wrap in userPrompt
    # adjust temperature to be lower? default 1
    response = model.chat.completions.create(
                    model='gpt-4-turbo', 
                    messages=[
                        {'role': 'system', 'content': sysPrompt},
                        {'role': 'user', 'content': userPrompt}
                    ]
                ).choices[0].message.content
    return response

def main(): 
    # ehr = "'''".join(getData(data_path))
    ehr = getData(data_path)
    sysPrompts = ['You are a medical professional', 'You are a neurologist', 'You are a medical professional specializing in neurology']
    userPrompts = ['Summarize the medical history', 'Summarize the neurological medical history', "Summarize the patient's medical history related to CAD"]

    results = []
    for sp in sysPrompts: 
        for up in userPrompts: 
            # summary = prompt(sp, up + "of the following reports separated by ''': '''" + ehr[0] + "'''" + ehr[1] + "'''")
            summary = prompt(sp, up + ' of the following report: ' + ehr[0])
            info = {
                'systemPrompt': sp,
                'userPrompt': up,
                'summary': summary
            }
            results.append(info)
    output(results, 'output-1record.json')