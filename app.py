import os
import time
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import gradio as gr
from charset_normalizer import detect
from chromadb.config import Settings
from epub2txt import epub2txt
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PDFMinerLoader,
    TextLoader,
)

# from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
import torch

# FAISS instead of PineCone
from langchain.vectorstores import  Chroma
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse

parser = argparse.ArgumentParser('LocalGPT falcon', add_help=False)
parser.add_argument('--device_type', type=str, default="cuda", choices=["cpu", "mps", "cuda"], help='device type', )
args = parser.parse_args()


ROOT_DIRECTORY = Path(__file__).parent
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/db"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)
ns = SimpleNamespace(qa=None)

# INSTRUCTORS_EMBEDDINGS_MODEL = "hkunlp/instructor-xl"
INSTRUCTORS_EMBEDDINGS_MODEL = "hkunlp/instructor-large"
# INSTRUCTORS_EMBEDDINGS_MODEL = "hkunlp/instructor-large"
# INSTRUCTORS_EMBEDDINGS_MODEL = "hkunlp/instructor-base"

def load_single_document(file_path: str or Path) -> Document:
    """ingest.py"""
    # Loads a single document from a file path
    # encoding = detect(open(file_path, "rb").read()).get("encoding", "utf-8")
    encoding = detect(Path(file_path).read_bytes()).get("encoding", "utf-8")
    if file_path.endswith(".txt"):
        if encoding is None:
            logger.warning(
                f" {file_path}'s encoding is None "
                "Something is fishy, return empty str "
            )
            return Document(page_content="", metadata={"source": file_path})

        try:
            loader = TextLoader(file_path, encoding=encoding)
        except Exception as exc:
            logger.warning(f" {exc}, return dummy ")
            return Document(page_content="", metadata={"source": file_path})

    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif Path(file_path).suffix in [".docx"]:
        try:
            loader = Docx2txtLoader(file_path)
        except Exception as exc:
            logger.error(f" {file_path} errors: {exc}")
            return Document(page_content="", metadata={"source": file_path})
    elif Path(file_path).suffix in [".epub"]:  # for epub? epub2txt unstructured
        try:
            _ = epub2txt(file_path)
        except Exception as exc:
            logger.error(f" {file_path} errors: {exc}")
            return Document(page_content="", metadata={"source": file_path})
        return Document(page_content=_, metadata={"source": file_path})
    else:
        if encoding is None:
            logger.warning(
                f" {file_path}'s encoding is None "
                "Likely binary files, return empty str "
            )
            return Document(page_content="", metadata={"source": file_path})
        try:
            loader = TextLoader(file_path)
        except Exception as exc:
            logger.error(f" {exc}, returnning empty string")
            return Document(page_content="", metadata={"source": file_path})

    return loader.load()[0]

def greet(name):
    """Test."""
    logger.debug(f" name: [{name}] ")
    return "Hello " + name + "!!"


def upload_files(files):
    """Upload files."""
    try:
        file_paths = [file.name for file in files]
    except:
        file_paths = [files]
    logger.info(file_paths)

    res = ingest(file_paths)
    logger.info("Processed:\n{res}")
    del res

    ns.qa = load_qa()

    return file_paths


def ingest(
    file_paths: list
):
    """Gen Chroma db.
    torch.cuda.is_available()
    file_paths =
    []
    """
    logger.info("Doing ingest...")

    documents = []
    for file_path in file_paths:
        documents.append(load_single_document(f"{file_path}"))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    logger.info(f"Loaded {len(documents)} documents ")
    logger.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    logger.info(f"Load InstructEmbeddings model: {INSTRUCTORS_EMBEDDINGS_MODEL}")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=INSTRUCTORS_EMBEDDINGS_MODEL, model_kwargs={"device": args.device_type}
    )

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None
    logger.info("Done ingest")

    return [
        [Path(doc.metadata.get("source")).name, len(doc.page_content)]
        for doc in documents
    ]


# https://huggingface.co/tiiuae/falcon-7b-instruct
def gen_local_llm():
    """Gen a local llm.
    localgpt run_localgpt
    """
    model = "tiiuae/falcon-7b-instruct"

    if args.device_type == "cuda":
        tokenizer = AutoTokenizer.from_pretrained(model)
    else: # cpu
        tokenizer=AutoTokenizer.from_pretrained(model)
        model=AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float32 if args.device_type =="cpu" else torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu" if args.device_type =="cpu" else "auto",
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


def load_qa():
    """Gen qa."""
    logger.info("Doing qa")

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=INSTRUCTORS_EMBEDDINGS_MODEL, model_kwargs={"device": args.device_type}
    )
    # xl 4.96G, large 3.5G,
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    llm = gen_local_llm()  # "tiiuae/falcon-7b-instruct"

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    logger.info("Done qa")

    return qa


def main1():
    """Lump codes"""
    with gr.Blocks() as demo:
        iface = gr.Interface(fn=greet, inputs="text", outputs="text")
        iface.launch()

    demo.launch()


def main():
    """Do blocks."""
    logger.info(f"ROOT_DIRECTORY: {ROOT_DIRECTORY}")

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Accordion("Info", open=False):
            _ = """
                Talk to your docs (.pdf, .docx, .csv, .txt .md). It
                takes quite a while to ingest docs (10-30 min. depending
                on net, RAM, CPU etc.).
                """
            gr.Markdown(dedent(_))
        title = """
            <div style="text-align: center;">
                <h1>LocalGPT with Falcon</h1>
                <p style="text-align: center;">Upload your docs (.pdf, .docx, .csv, .txt .md) by clicking the "Load docs to LangChain" and wait until the upload is complete, <br />
                when everything is ready, you can start asking questions about the docs <br />
            </div>
        """
        gr.HTML(title)
        with gr.Tab("Upload files"):
            # Upload files and generate embeddings database
            file_output = gr.File()
            upload_button = gr.UploadButton(
                "Load docs to LangChain",
                file_count="multiple",
            )
            upload_button.upload(upload_files, upload_button, file_output)

            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Query")
            clear = gr.Button("Clear")

            def respond(message, chat_history):
                if ns.qa is None:  # no files processed yet
                    bot_message = "Provide some file(s) for processsing first."
                    chat_history.append((message, bot_message))
                    return "", chat_history
                try:
                    res = ns.qa(message)
                    answer, docs = res["result"], res["source_documents"]
                    bot_message = f"{answer}"
                except Exception as exc:
                    logger.error(exc)
                    bot_message = f"bummer! {exc}"

                chat_history.append((message, bot_message))

                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

    try:
        from google import colab

        share = True  # start share when in colab
    except Exception:
        share = False

    demo.launch(share=share)


if __name__ == "__main__":
    main()
