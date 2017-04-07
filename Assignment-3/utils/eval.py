#!/usr/bin/python3

# Reference: https://www.cyberciti.biz/faq/python-run-external-command-and-get-output/

import subprocess

def main():
	systemPOS = "berp-out.txtdata_test_{0}.txt"
	goldPOS = "data/berp-key_{0}.txt"

	acc = []
	for i in range(1, 30):
		result = systemPOS.format(i)
		expected = goldPOS.format(i)

		# Create a command line to run the evalPosTagger.py 
		cmd = ("{cmd} {pythonFile} {goldPOS_filename} {systemPOS_filename}".
            format(cmd="python3", pythonFile="evalPOSTagger.py", 
                goldPOS_filename = expected, systemPOS_filename = result))

		# Read the output on the console 
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
		(output, err) = p.communicate()
		p_status = p.wait()
		output = str(output)

		# take a substring to extract the accuracy 
		acc.append(float(output[15:-3]))

	# Calculate the average 
	# print(acc)
	ave = sum(acc) / len(acc) 
	print("average accuracy:", ave)


		

if __name__ == '__main__':
	main()



