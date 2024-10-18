import subprocess
while(1):
    print("Enter 0: Exit 1: Heart 2: Diabeties")
    print("Enter Option: ")
    t=input()
    if(t=="0"): 
        break
    if(t=='1'):
        subprocess.run(["python","heart-disease.py"])
        print()
    elif(t=='2'):
        subprocess.run(["python","diabetes.py"])
        print()
    else:
        print("Invalid Input!\nChoose from given options")