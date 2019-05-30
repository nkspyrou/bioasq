#annotation of questions with pico elements as 0 or 1

questions = open("foreground_questions.txt", "r")
pico_annotated_questions = open("pico_annotated_questions.txt", "a+")
#format of pico annotated questions:
#foreground		question	p	i	c	o	t	d	p	h
#tdph = treatment diagnosis prognosis harm
for x in questions:
	print(x)
	p_input = input("p\n") 
	i_input = input("i\n") 
	c_input = input("c\n") 
	o_input = input("o\n") 
	type_input = input("t(herapy)  d(iagnosis)  p(rognosis)  h(arm)\n") 
	pico_annotated_questions.write(x.rstrip() + ("\t1\t" if p_input == "1" else "\t0\t" )+ ( "1\t" if i_input == "1" else "0\t") + \
	("1\t" if c_input == "1" else "0\t" )+ ("1\t" if o_input == "1" else "0\t")+  ("1\t" if type_input == "t" else "0\t") + \
	("1\t" if type_input == "d" else "0\t") + ("1\t" if type_input == "p"  else "0\t") + ("1\t" if type_input == "h" else "0\t") +"\n")