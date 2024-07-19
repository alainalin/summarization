import os 
import evaluate #from hugging face 
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

data_path = '/mnt/nfs/CanarySummarization/Data'

## models
# api key? mistral account? 
model = ''
client = MistralClient()

## NLP metrics
# poor correlation with human judgement, fallacy of references
bleu = evaluate.load('bleu') # completeness? 
rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore') # semantic
# medcon or use knowledge base to measure conceptual correctness

# experiment with different user and system prompts 
def promptModel(prompt: str) -> str:
    model_spec = 'You are a medical professional. The patient has the following medical history which includes clinical notes from previous hospital visits.' 
    # set behavior and context
    # add summarization instructions, knowledge graph derived relevant concepts

    messages = [ChatMessage(role='system', content=model_spec), ChatMessage(role='user', content=prompt)]

    response = client.chat(
        model=model, 
        messages=messages,
    )

    return response.choices[0].message.content



