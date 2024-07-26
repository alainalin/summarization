import os, json
import xml.etree.ElementTree as ET
import evaluate 

model_path = '/mnt/nfs/CanaryModels/Data'

# data_path = '/mnt/nfs/CanarySummarization/Data'
data_path = '/home/alaina/Documents/summarization/example_data_one_patient'

## NLP metrics
bleu = evaluate.load('bleu') # completeness 
rouge = evaluate.load('rouge') # most commonly used
bertscore = evaluate.load('bertscore') # semantic/correctness
# medcon or use knowledge base to measure conceptual correctness

def getData(path: str, start=0, stop=None) -> list[str]: 
    '''
    path :: directory path to the dataset to extract from
    start :: int, start getting reports at this index (inclusive, 0 index)
    stop :: int, stop getting reports at this index (exclusive, 0 index)

    return data from [start, stop) indices
    '''
    reports = []
    listdir = os.listdir(dataset)

    if stop is None: 
        stop = len(listdir)

    if stop < start or start >= len(listdir) or stop <= 0 or stop > len(listdir): 
        raise ValueError('Invalid bounds to fetch data') 

    for i in range(start, stop):
        with open(os.path.join(dataset, listdir[i]), 'r', encoding='utf-8') as file:
            xml_text = file.read()
            root = ET.fromstring(xml_text)
            text_content = root.find(".//TEXT").text
            reports.append(text_content)
            
    return reports

def output(data, file):
    '''
    output data into specified file
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
    # update(results, 'output-2records.json')
    output(results, 'output-1record.json')

main()

