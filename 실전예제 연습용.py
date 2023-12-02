# 모험가 길드
"""
n=int(input())
data=list(map(int,input().split()))

data.sort()

count=0
result=0

for i in data:
    count+=1
    if count>=i:
        result+=1
        count=0

print(result)
"""
# 곱하기 혹은 더하기
"""
s=input()
data=[]

for i in range(len(s)):
    data.append(int(s[i]))

result=data[0]

for i in range(1,len(data)):
    if data[i]<=1 or result<=1:
        result+=data[i]
    else:
        result*=data[i]

print(result)
"""
# 문자열 뒤집기
"""
s=input()
count0=0
count1=0

if s[0]==1:
    count0+=1
else:
    count1+=1
    
for i in range(len(s)-1):
    if s[i]!=s[i+1]:
        if s[i+1]==1:
            count0+=1
        else:
            count1+=1
print(min(count0,count1))
"""
# 만들 수 없는 금액
"""
n=int(input())
data=list(map(int,input().split()))

data.sort()

target=1

for i in data:
    if target<i:
        break
    target+=i

print(target)
"""
# 볼링공 고르기
"""
from itertools import combinations

n,m=map(int, input().split())
data=list(map(int,input().split()))

result=list(combinations(data,2))

count=0
for i in result:
    if i[0]==i[1]:
        continue
    else:
        count+=1

print(count)
"""
# 럭키 스트레이트
"""
n=int(input())

r1=0
r2=0
d=list(str(n))

for i in range(len(d)//2):
    r1+=int(d[i])
for i in range(len(d)//2,len(d)):
    r2+=int(d[i])

if r1==r2:
    print("LUCKY")
else:
    print("READY")

"""
# 문자열 재정렬
"""
data=input()
result=[]
summ=0

for x in data:
    if x.isalpha():
        result.append(x)
    else:
        summ+=int(x)
result.sort()
result.append(str(summ))

print(''.join(result))
"""
#문자열 압축(내가 푼 것x)
"""
def solution(s):
    answer=len(s)
    for step in range(1,len(s)//2+1):
        compressed=""
        prev=s[0:step]
        count=1
        for j in range(step,len(s),step):
            if prev==s[j:j+step]:
                count+=1
            else:
                compressed+=str(count)+prev if count >= 2 else prev
                prev=s[j:j+step]
                count=1
        compressed+=str(count)+prev if count>=2 else prev
        answer=min(answer, len(compressed))
    return answer

data="aabbaccc"
print(solution(data))
"""
# 뱀




























