import xlrd,math
from sklearn.metrics import confusion_matrix
unknown_vector=list();main_list=list();y_actu=list();y_pred=list()
inside_sum=0;sum=0;accuracy_count=0
sample_class_count=input("No. of samples class-wise in training samples: ")         #check the data set that each class contains this many samples
no_of_class=input("No of classes overall: ")
workbook=xlrd.open_workbook("samp.xlsx")                                            #contains extrated features with first row ie '0' containing function name and function 1 starts from col '0'
sheet=workbook.sheet_by_index(0)
row_count=sheet.nrows
col_count=sheet.ncols
for alpha in range(no_of_class*sample_class_count+1,row_count):
    for c in range(0,col_count-1):
        unknown_vector.append(sheet.cell_value(alpha,c))
    unknown_vector.append(sheet.cell_value(alpha,col_count))
    for row in range(1,no_of_class*sample_class_count+1):
        cool=0
        for col in range(0,col_count-1):
            inside_sum+=(sheet.cell_value(row,col)-unknown_vector[cool])**2
            cool+=1
        sum+=math.exp(-(inside_sum)/(2.0))
        inside_sum=0
        if row%sample_class_count==0:
            main_list.append((1/float(sample_class_count))*sum)
            sum=0
    print(main_list)
    y_pred.append(main_list.index(max(main_list))+1)
    print("Maximum value belongs to Class"+str(main_list.index(max(main_list))+1))
    if main_list.index(max(main_list))+1==unknown_vector[len(unknown_vector)-1]:
        accuracy_count+=1
    else:
        print("Class identification mismatch for",unknown_vector)
    del main_list[:]
    del unknown_vector[:]
accuracy_count=(accuracy_count/50.0)                #50 test samples
print ("Accuracy=",accuracy_count)
#return accuracy_count
for row in range(no_of_class*sample_class_count+1,row_count):
    y_actu.append(sheet.cell_value(row,col_count))
print "Confusion Matrix: "
print(confusion_matrix(y_actu, y_pred))
