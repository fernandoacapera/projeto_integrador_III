from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch

titulo = "🤖ChatBot de IA"
descricao = "Construir chatbots de domínio aberto é uma área desafiadora para a pesquisa em aprendizado de máquina."
exemplos = [["Como você está?"]]


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
modelo = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")


def prever(entrada, historico=[]):
    novos_tokens_usuario = tokenizer.encode(
        entrada + tokenizer.eos_token, return_tensors="pt"
    )
    tokens_entrada_bot = torch.cat([torch.LongTensor(historico), novos_tokens_usuario], dim=-1)
    historico = modelo.generate(
        tokens_entrada_bot, max_length=4000, pad_token_id=tokenizer.eos_token_id
    ).tolist()
    resposta = tokenizer.decode(historico[0]).split("<|endoftext|>")
    resposta = [
        (resposta[i], resposta[i + 1]) for i in range(0, len(resposta) - 1, 2)
    ]
    return resposta, historico
gr.Interface(
    fn=prever,
    title=titulo,
    description=descricao,
    examples=exemplos,
    inputs=["text", "state"],
    outputs=["chatbot", "state"], 
    theme="finlaymacklon/boxy_violet",
).launch()
