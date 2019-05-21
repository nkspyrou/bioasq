import json

question_dict = {}
foreground_background_pandata = open("C:\\Users\\user\\Desktop\\foreground_background_pandata.txt" , "a+")

with open("C:\\Users\\user\\Desktop\\trainining7b.json") as json_file:  
	data = json.load(json_file)
	for p in data['questions']:
		question_dict[p['id']] = p['body']
		
picos =  open("C:\\Users\\user\\Desktop\\pico_questions.txt", 'r')
other =  open("C:\\Users\\user\\Desktop\\other_medical_questions.txt", 'r')

for x in picos:
	foreground_background_pandata.write("foreground\t" + question_dict[x.rstrip()] + "\n")
	
	
for x in other:
	foreground_background_pandata.write("background\t"+ question_dict[x.rstrip()] + "\n")
	