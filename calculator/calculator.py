import pickle as pkl
import tkinter as tk

MAIN_WIDTH = 1000
MAIN_HEIGHT = 500
BASE_HEIGHT = 100

def clf_predict(entry_obj_list, clf, canvas, label):
	# NOTE: Order of tkinter fields is not the same order as model input

	label['text'] = ''
	order = ['Urea Nitrogen', 'Blood Albumin', 'Lactate Dehydrogenase', 'Blood pH', 'Age', 
			'History of Chronic Cardiac disease (not hypertension)',
			'Oxygen saturation', 'Oxygen saturation measured on',
			'Number of medications']    
	vars_ = [None] * len(entry_obj_list)

	# Get information from fields
	for i, var in enumerate(order):
		for obj in entry_obj_list:
			if var==obj[0][0]:
				try:
					field_value = obj[1].get()
					# If any value convert to float, else raise error
					if len(field_value) == 0:
						continue
					else:
						vars_[i] = float(field_value)
				except ValueError:
					label['text'] = 'Please fill fields with numerical values'
					return

	if vars_[7] == 0:
		vars_.insert(7, 1.0)
	else:
		vars_.insert(7, 0.0)
	print([vars_])
	# vars_ = [None] * 10
	y_hat = clf.predict_proba([vars_])
	label['text'] = 'Predicted chance of death = {:.3f}'.format(y_hat[0][1])



if __name__=='__main__':
	with open('xgb_classifier.pkl', 'rb') as f:
		clf = pkl.load(f)

	root = tk.Tk()
	root.title('COVID19 calculator')
	canvas = tk.Canvas(root, width=MAIN_WIDTH, height=MAIN_HEIGHT)
	canvas.pack()

	# ENTRY BOXES
	entries = [('Age', 'years'), 
			('Number of medications', 'count'),
			('History of Chronic Cardiac disease (not hypertension)', 'No=0 | Yes=1'),
			('Blood pH', 'pH'),
			('Blood Albumin', 'g/L'),
			('Urea Nitrogen', 'mmol/L'),
			('Lactate Dehydrogenase', 'mmol/L'),
			('Oxygen saturation', '%'),
			('Oxygen saturation measured on', 'Room air=0 | Oxygen therapy=1')]

	entry_obj_list = []
	for i, entry in enumerate(entries):
		current_entry_obj = tk.Entry(root)
		entry_obj_list.append((entry, current_entry_obj))
		canvas.create_window(MAIN_WIDTH/2, BASE_HEIGHT+i*22, window=current_entry_obj)
		label = tk.Label(root, text=entry[0])#, anchor='e', justify='right')
		label.config(font=('helvetica', 12))
		canvas.create_window(MAIN_WIDTH/2-70, BASE_HEIGHT+i*22, window=label, anchor='e')
		label_unit = tk.Label(root, text=entry[1])
		label_unit.config(font=('helvetica', 10))
		canvas.create_window(MAIN_WIDTH/2+70, BASE_HEIGHT+i*22, window=label_unit, anchor='w')

	# Output label
	label = tk.Label(root, text='')
	label.config(font=('helvetica', 14))
	canvas.create_window(MAIN_WIDTH/2, BASE_HEIGHT+len(entry_obj_list)*20+70, window=label)

	# BUTTON
	button = tk.Button(text='Get prediction', bg='brown', fg='white', 
					command=lambda: clf_predict(entry_obj_list, clf, canvas, label))
	canvas.create_window(MAIN_WIDTH/2, BASE_HEIGHT+len(entry_obj_list)*20+30, window=button)

	# WARNING
	label_warn = tk.Label(root, text='!! WARNING !!\nThis tool is NOT VALIDATED and should ONLY be used for research purposes!', fg='white', bg='red')
	label_warn.config(font=('helvetica', 16))
	canvas.create_window(MAIN_WIDTH/2, 50, window=label_warn)

	# GO
	root.mainloop()