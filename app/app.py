import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


MODEL_PATH = "Mathking/bert-base-german-cased-gnad10"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
pipe = pipeline("text-classification", tokenizer=tokenizer,
                model=model, return_all_scores=True)

def predict(news):
    res_dict = {}
    for object in pipe(news)[0]:
        res_dict[object['label']] = object['score']
    return res_dict

inputs = gr.inputs.Textbox(lines=10,placeholder="Paste German article Here",label="Article")
outputs = gr.outputs.Label(num_top_classes=3,type="auto",label="News categories probabilities")
title = "Demo : German 📰News Classification !"
description = "A German BERT model trained on GNAD10 german news dataset, juste paste a german article in the Article input and press Submit within a little time the predictions probabilities will appear on the right output. The different news categories that the model can predict are : Web, Panorama, International, Wirtschaft, Sport, Inland, Etat, Wissenschaft, Kultur"
article = "<p style='text-align: center'><a href='https://www.deepset.ai/german-bert'>Deepset German BERT model</a> | <a href='https://github.com/mathieulaiking/10k-german-news-classification'>Github Repo</a></p>"
examples = ["Erfundene Bilder zu Filmen, die als verloren gelten: The Forbidden Room von Guy Maddin und Evan Johnson ist ein surrealer Ritt durch die magischen Labyrinthe des frÃ¼hen Kinos. Wien â€“ Die Filmgeschichte ist ein Friedhof der Verlorenen. Unter den Begrabenen finden sich zahllose Filme, von denen nur noch mysteriÃ¶s oder abenteuerlich klingende Namen kursieren","21-JÃ¤hriger fÃ¤llt wohl bis Saisonende aus. Wien â€“ Rapid muss wohl bis Saisonende auf Offensivspieler Thomas Murg verzichten. Der im Winter aus Ried gekommene 21-JÃ¤hrige erlitt beim 0:4-Heimdebakel"]

iface = gr.Interface(fn=predict,inputs=inputs,outputs=outputs,title=title,description=description,article=article,examples=examples,server_name="0.0.0.0",)
iface.launch()
