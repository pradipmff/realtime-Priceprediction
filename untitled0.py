x= S
num= K
lst1 = list(x)
if 1<= len(lst1)<=103:
    if 1<= num<= 106:
        for i in range(num):
            
            for i in range(len(lst1)) :
                if lst1[i] == "R" :
                    lst1[i] = "G"
                    
                elif lst1[i] == "G" :
                     lst1[i] = "B"
                     
                elif lst1[i] == "B" :
                     lst1[i] = "R"
p=""
result= p.join(lst1)