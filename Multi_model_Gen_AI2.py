from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, get_peft_model, LoraConfig
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from diffusers import StableDiffusionPipeline
import torch
import streamlit as st
from torch.cuda.amp import autocast
from safetensors.torch import load_file
from torch.cuda.amp import autocast
import streamlit as st
from torch import autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#LoRA Text
text_model_dir = "........."

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
lora_model = PeftModel.from_pretrained(base_model, text_model_dir, local_files_only = True).to(device = device)


#LoRA Image
device = 'cuda'
sd_lora_dir = ".........."


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16  
)


pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    sd_lora_dir 
)


pipe.unet.to(device)
pipe.vae.to(device)
pipe.text_encoder.to(device)


pipe.safety_checker = lambda images, **kwargs: (images, [False]*len(images))

#RAG
rag_dir = ".........."

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs = {"device" : device})
vectorstore = FAISS.load_local(rag_dir, embedding_model, allow_dangerous_deserialization = True)
pipe_lm = pipeline("text-generation", model=lora_model, tokenizer=tokenizer, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id,
                   do_sample = True, temperature = 0.7)
llm = HuggingFacePipeline(pipeline=pipe_lm)

template = """
Based on the context below, answer the question in one concise sentence.
If you don't know the answer, say "I don't know."

Context:
{context}

Question:
{question}

Helpful Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)




qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(),
                                 return_source_documents=True, chain_type="stuff",
                                 chain_type_kwargs={"prompt": prompt})


# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "qa" not in st.session_state:
    st.session_state.qa = qa  

if "pipe" not in st.session_state:
    st.session_state.pipe = pipe  

# Multi-modal assistant function
def multi_model_assistant(query):

    if query.lower().startswith(("generate image of", "generate an image", "generate image")):
        prompt = query
        with autocast(device_type = 'cuda', dtype=torch.float16):
            image = st.session_state.pipe(
                prompt=prompt,
                guidance_scale=7.5,
                num_images_per_prompt=1,
            ).images[0]
        return "", image
    
    rag_result = st.session_state.qa({"query": query})
    text_answer = rag_result['result']
    if "Helpful Answer:" in text_answer:
        text_answer = text_answer.split("Helpful Answer:")[-1].strip()
    else:
        text_answer = text_answer.strip()
    return text_answer, None

# Submit button
with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything:")
    submit = st.form_submit_button("Submit")

if submit and user_input:
    answer, image = multi_model_assistant(user_input)
    st.session_state.history.append({
        "user": user_input,
        "assistant": answer,
        "image": image
    })

# Display chat history
for chat in st.session_state.history:
    st.write("You : ", chat['user'])
    st.write("Assistant : ", chat['assistant'])
    if chat['image'] is not None:
        st.image(chat['image'], use_container_width=True)