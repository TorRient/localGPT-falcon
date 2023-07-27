from langchain.chains import RetrievalQA
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import click
import torch
from constants import CHROMA_SETTINGS


def load_model(device):
    """
    Select a model on huggingface.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.
    """
    model = "tiiuae/falcon-7b-instruct"

    if device == "cuda":
        tokenizer = AutoTokenizer.from_pretrained(model)
    else: # cpu
        tokenizer=AutoTokenizer.from_pretrained(model)
        model=AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float32 if device =="cpu" else torch.bfloat16,
        trust_remote_code=True,
        device_map=device if device =="cpu" else "auto",
        max_length=2048,
        temperature=0,
        top_p=0.95,
        top_k=10,
        repetition_penalty=1.15,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm


# @click.command()
# @click.option('--device_type', default='gpu', help='device to run on, select gpu or cpu')
# def main(device_type, ):
#     # load the instructorEmbeddings
#     if device_type in ['cpu', 'CPU']:
#         device='cpu'
#     else:
#         device='cuda'
 
    
 ## for M1/M2 users:

@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
def main(device_type, ):
    # load the instructorEmbeddings
    if device_type in ['cpu', 'CPU']:
        device='cpu'
    elif device_type in ['mps', 'MPS']:
        device='mps'
    else:
        device='cuda'

    print(f"Running on: {device}")

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base",
                                               model_kwargs={"device": device})
    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # Prepare the LLM
    # callbacks = [StreamingStdOutCallbackHandler()]
    # load the LLM for generating Natural Language responses. 
    llm = load_model(device)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        print("----------------------------------SOURCE DOCUMENTS---------------------------")
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
        print("----------------------------------SOURCE DOCUMENTS---------------------------")


if __name__ == "__main__":
    main()
