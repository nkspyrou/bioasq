import json
import os

#JSON parser that iterates over all bioasq datasets and detects the yes/no questions that contain words pertinent to oncology
f= open("already_classified_questions.txt","a+") #file that counts where we have stopped

#USEFUL LISTS/DICTS
foreground_list = list() #categories of questions that contain q ids
background_list = list()
other_list = list()
already_classified_list = list()
question_dict = {}

read_classified = open("already_classified_questions.txt", "r")
for x in read_classified:
  already_classified_list.append(x.rstrip())

 

with open("C:\\Users\\user\\Desktop\\trainining7b.json") as json_file:  
	data = json.load(json_file)
	for p in data['questions']:
		if  (p['id'] not in already_classified_list): #and  p['type'] == "yesno" and (("cancer" in p['body']) or ("tumor" in p['body'])  or ("metastasis" in p['body'])or ("oncology" in p['body'])):
			print('q: ' + p['body'])
			question_dict[p['id']] = (p['body'] , p['id'] , p['body'])
			category_input = input("Categorize: 1/intervention, 2/correlation, 3/other\n") 
			if category_input == "1":
				foreground_list.append(p['id'])
				f.write(p['id'] + "\n")
			elif category_input == "2":
				background_list.append(p['id'])
				f.write(p['id'] + "\n")
			elif category_input == "3":
				other_list.append(p['id'])
				f.write(p['id'] + "\n")
			# else:
				# break
	

			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
# for filename in os.listdir("C:\\Users\\user\\Desktop\\BioASQ-TaskB-testData"):
	# if filename.endswith('.json'): 
		# with open("C:\\Users\\user\\Desktop\\BioASQ-TaskB-testData\\" + filename) as json_file:  
			# print(filename)
			# data = json.load(json_file)
			# for p in data['questions']:
				# if p['type'] == "yesno" and (("cancer" in p['body']) or ("tumor" in p['body'])  or ("metastasis" in p['body'])or ("oncology" in p['body'])):
					# print('q: ' + p['body'])
					# question_dict[p['id']] = (p['body'] , p['id'] , p['body'])
					# category_input = input("Categorize: 1/intervention, 2/correlation, 3/other") 
					# if category_input == "1":
						# intervention_list.append(p['id'])
					# elif category_input == "2":
						# correlation_list.append(p['id'])
					# elif category_input == "3":
						# other_list.append(p['id'])
				# print(intervention_list)
			# print("\n")
# print(intervention_list)





