import math
n,m=list(map(int,input().split()))
array=[True for i in range(10000001)]
array[1]=0
for i in range(2,int(math.sqrt(m))+1):
    if array[i]==True:
        j=2
        while i*j<=m:
            array[i*j]=False
            j+=1
for k in range(n,m+1):
    if array[i]:
        print(i)
