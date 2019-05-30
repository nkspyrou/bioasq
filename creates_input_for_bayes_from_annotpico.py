import pandas as pd

read_pico = open('C:\\Users\\user\\Desktop\\pico_annotated_questions.txt', "r")
newpicoannot = open('C:\\Users\\user\\Desktop\\newpicoannot.txt', "a+")
# labels = list()
# messages = list()
for x in read_pico:
  x = x.split('\t')
  # messages.append(x[1])
  # labels.append(x[7])
  newpicoannot.write(x[7] + "\t" + x[1] + "\t"  + x[2] + "\n")
				
  
# print(labels)
 
df = pd.read_csv('C:\\Users\\user\\Desktop\\newpicoannot.txt',  
                   sep='\t', 
				   
                   header=None,
                   names=['label','message', 'code'])

print(df['label'])