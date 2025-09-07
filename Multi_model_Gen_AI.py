import torch 
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
import os, numpy as np, faiss, re
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.cuda.amp import autocast
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import streamlit as st

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LoRA fine-tuning of text generation

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

dataset = load_dataset("blended_skill_talk", split = "train[:1000]")

def preprocess(data):
    text = data['previous_utterance'][-1] + " " + data['free_messages'][0]
    tokenized = tokenizer(text, truncation = True, max_length = 128, padding = "max_length")
    tokenized['labels'] = tokenized['input_ids']
    return tokenized

training_data = dataset.map(preprocess, remove_columns = dataset.column_names)
training_data.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])


lora_config = LoraConfig(task_type = "CAUSAL_LM",
                         r = 16, lora_alpha = 32, lora_dropout = 0.1, target_modules=["q_proj", "v_proj"])

lora_model = get_peft_model(model = model, peft_config = lora_config).to(device)
lora_model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir = r'C:\Users\kaifm\OneDrive\Desktop\Machine_Learning\Generative_AI\flan_t5_lora_output',
    num_train_epochs = 10,
    per_device_train_batch_size = 2, 
    logging_steps = 10,
    save_strategy = "epoch",
    learning_rate = 1e-4 ,
    fp16 = True
)


trainer = Trainer(model = lora_model, args = training_args, train_dataset = training_data)
trainer.train()

lora_model.eval()
prompt = "Hello, what is your name?"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = lora_model.generate(
    inputs["input_ids"].to(device),
    max_length=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)

reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Bot:", reply)


# LoRA stable diffusion

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                               torch_dtype = torch.float16).to(device)

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 32, 
    lora_dropout = 0.1,
    target_modules = ["to_q", "to_v"],
    bias = "none",
)

dataset = load_dataset("beans", split = "train[:300]")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

pipe.unet = get_peft_model(model = pipe.unet, peft_config = lora_config)

def collate_fn(batch):
    images = [transform(x['image']) for x in batch]
    labels = [x['labels'] for x in batch]
    prompt = [f"A bean plant leaf {dataset.features['labels'].int2str(label)} disease" for label in labels]
    return torch.stack(images), prompt

train_loader = torch.utils.data.DataLoader(dataset = dataset, shuffle = True,
                                           batch_size = 1, collate_fn = collate_fn)

vae = pipe.vae.to(device)
unet = pipe.unet
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer

noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                subfolder = "scheduler")
optimizer = torch.optim.AdamW(params = unet.parameters(), lr = 1e-4)

for epoch in range(10):
    running_loss = 0.0
    for images, prompts in train_loader:
        images = images.to(device)
        with autocast(dtype = torch.float16):
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, 
                                      (latents.shape[0],), device = device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            inputs = tokenizer(prompts, padding = "max_length", truncation = True, 
                               return_tensors = "pt").to(device)
            encoder_hidden_states = text_encoder(inputs.input_ids)[0]
            pred_noise = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        error = running_loss/len(train_loader)
    print(f"Epoch : {epoch + 1} ---- Loss : {error:.4f}")


pipe.unet = unet
pipe.vae = vae
pipe = pipe.to(device)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

#RAG (Retrieval Augmented Generation)

dataset = load_dataset("blended_skill_talk", split = "train")
docs = []
for info in dataset:
    context_text = info['previous_utterance'][-1] + " " + info['free_messages'][0]
    docs.append(Document(page_content = context_text))

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(docs, embedding_model)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = lora_model
pipeline = pipeline("text-generation", model = model, tokenizer = tokenizer, max_new_tokens = 50, 
                pad_token_id = tokenizer.eos_token_id)
llm = HuggingFacePipeline(pipeline = pipeline)

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

qa = RetrievalQA.from_chain_type(llm = llm, retriever = vectorstore.as_retriever(),
                                 return_source_documents = True, chain_type = "stuff",
                                 chain_type_kwargs={"prompt": prompt})



save_dir = r".........."
lora_model.save_pretrained(save_directory = save_dir)
tokenizer.save_pretrained(save_dir)
save_dir = r".........."
vectorstore.save_local(save_dir)
print("saved Eleuther LoRA model and tokenizer", save_dir)


save_dir = r"..........."
pipe.unet.save_pretrained(save_dir)
print("Saved Stable-Diffusion", save_dir)