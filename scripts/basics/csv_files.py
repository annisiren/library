# CSV FILES

import csv


# READ
data = open('example.csv',encoding="utf-8")
csv_data = csv.reader(data)
data_lines = list(csv_data)

# WRITE

# newline controls how universal newlines works (it only applies to text
# mode). It can be None, '', '\n', '\r', and '\r\n'.
file_to_output = open('to_save_file.csv','w',newline='')
csv_writer = csv.writer(file_to_output,delimiter=',')

csv_writer.writerows([['1','2','3'],['4','5','6']])
file_to_output.close()

# WRITE TO EXISTING FILE

f = open('to_save_file.csv','a',newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['new','new','new'])
f.close()



##################
# PDF FILES
# note the capitalization
import PyPDF2

# Notice we read it as a binary with 'rb'
f = open('Working_Business_Proposal.pdf','rb')

pdf_reader = PyPDF2.PdfFileReader(f)
