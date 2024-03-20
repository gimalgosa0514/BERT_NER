import torch
import torchmetrics
import plotly.figure_factory as ff
#csv를 읽어들일 때 리스트를 문자열로 읽어들이는 문제 해결
from ast import literal_eval
import pandas as pd
from seqeval.metrics import classification_report

def str_to_list(x):
  try:
    if type(x) == str:
      return literal_eval(x)
    elif type(x) == list:
      return x
  except:
    return None

df = pd.read_csv("test_result.csv",encoding="utf-8")
df = df[df["labels"] != "labels"]

df["labels"] = df["labels"].apply(lambda x: str_to_list(x))
df["preds"] = df["preds"].apply(lambda x: str_to_list(x))

labels = df["labels"].tolist()
preds = df["preds"].tolist()

label = []
pred = []


for i in range(len(labels)):
  if labels[i] == None:
    break
  else:
    for ref, predd in zip(labels[i],preds[i]):
      label.append(ref)
      pred.append(predd)

true = [label]
predicted = [pred]


f1 = torchmetrics.F1Score(task = "multiclass", num_classes = 13, average ="macro")
label_str_to_id = lambda s:{'B-DT':0, 'I-DT':1, 'B-LC':2, 'I-LC':3, 'B-OG':4, 'I-OG':5, 'B-PS':6, 'I-PS':7, 'B-QT':8, 'I-QT':9, 'B-TI':10, 'I-TI':11, 'O':12}[s]

preds = torch.tensor(list(map(label_str_to_id,pred)))
labels = torch.tensor(list(map(label_str_to_id,label)))

f1_score = f1(preds, labels)
acc = torch.mean((preds == labels).float())


result = f"f1_score : {f1_score}" + "\n" + f"acc : {acc}" + "\n" + '\n'
#결과 파일에 쓰기
result = result + "class" +classification_report(true,predicted)
f = open("result.txt","w")
f.write(result)
f.close()

confusion_func = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=13)
confusion_matrix = confusion_func(preds, labels)
label = ['B-DT', 'I-DT', 'B-LC', 'I-LC', 'B-OG', 'I-OG', 'B-PS', 'I-PS', 'B-QT', 'I-QT', 'B-TI', 'I-TI', 'O']

fig = ff.create_annotated_heatmap(confusion_matrix.numpy(), label, label)
fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="True Label")
fig.write_image("confusion_matrix.png", format="png", scale=2)

