# heapq 함수를 사용하여 heap정렬 구현하기
'''
import heapq

def heapsort(iterable):
    h=[]
    result=[]
    for value in iterable:
        heapq.heappush(h,value)

    for _ in range(len(h)):
        result.append(heapq.heappop(h))
    return result
result=heapsort([1,3,5,7,9,2,4,6,8,0])
print(result)
'''
# 거스름돈 문제
"""
n = int(input("손님이 낸 돈?: "))
count = 0

while n>=10:
    if n >= 500:
        count += n//500
        n %= 500

    elif n >= 100:
        count += n//100
        n %= 100

    elif n >= 50:
        count += n//50
        n %= 50
    
    elif n >= 10:
        count += n//10
        n %= 10
print(count)
"""
# 그리디(큰 수의 법칙) - p.92
"""
n, m, k  = map(int, input().split())
data = list(map(int, input().split()))

data.sort()
first = data[n-1]
second = data[n-2]
result = 0

while True:
    for _ in range(k):
        if m==0:
            break
        result += first
        m-=1    
    if m==0:
        break
    result+=second
    m-=1
print(result)
"""
# 숫자 카드 게임
"""
n, m = map(int, input().split())

result = 0

for _ in range(n):
    data = list(map(int, input().split()))
    min_value = min(data)
    result = max(result, min_value)

print(result)
"""
# 1이 될 때까지
"""
n, k = map(int, input().split())
result = 0

while True:
    while n % k != 0:
        n -= 1
        result += 1
    n = n//k
    result += 1
    if n < k:
        break
while n > 1:
    n -= 1
    result += 1

print(result)
"""
# 상하좌우
"""
n = int(input())
data = input().split()
x, y = 1, 1

dx = [0,0,-1,1]
dy = [-1,1,0,0]
dset = ['L','R','U','D']

for i in data:
    for j in range(len(dset)):
        if i == dset[j]:
            x += dx[j]
            y += dy[j]
    if x<1 or y<1 or x>n or y>n:
        continue
print(x,y)
"""
# 시각
"""
n = int(input())

result = 0
for i in range(n+1):
    for j in range(60):
        for k in range(60):
            if '3' in str(i)+str(j)+str(k):
                result+=1

print(result)
"""
# 팩토리얼
"""
def factorial(n):
    result = 1
    for i in range(1,n+1):
        result*=i
    return result

def factorial_2(n):
    if n<=1:
        return 1
    return n*factorial_2(n-1)
print(factorial(5))
print(factorial_2(5))
"""
# DFS
"""
def dfs(graph,v,visited):
    visited[v]=True
    print(v,end=' ')

    for i in graph[v]:
        if not visited[i]:
            dfs(graph,i,visited)
"""
# BFS
"""
from collections import deque

def dfs(graph,start,visited):
    queue = deque([start])
    visited(start)=True

    while queue:
        v = queue.popleft()
        print(v,end=' ')

        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited(i)=True
"""
# 음료수 얼려먹기
"""
n,m=map(int,input().split())

graph=[]
for i in range(n):
    graph.append(list(map(int,input().split())))

def dfs(x,y):
    if x<-1 or x>n or y<-1 or y>n:
        return False
    if graph[x][y]==0:
        graph[x][y]==1
        dfs(x-1,y)
        dfs(x,y-1)
        dfs(x+1,y)
        dfs(x,y+1)
        return True
    return False

result=0
for i in range(n):
    for j in range(m):
        if dfs[i][j]==True:
            result += 1
print(result)
"""
# 미로 탈출
"""
from collections import deque

n,m = map(int,input().split())
graph=[]
for i in range(n):
    graph.append(list(map(int,input().split())))

dx=[-1,1,0,0]
dy=[0,0,-1,1]

def bfs(x,y):
    queue=deque()
    queue.append((x,y))

    while queue:
        x,y=queue.popleft()
        for i in range(4):
            nx=x+dx[i]
            ny=y+dy[i]

            if graph[nx][ny]==1:
                graph[nx][ny]=graph[x][y]+1
                queue.append((nx,ny))
    return graph[n-1][m-1]
"""
# 퀵 정렬
"""
array = [5,7,9,0,3,1,6,2,4,8]

def quick_sort(array, start, end):
    if start>=end:
        return
    pivot=start
    left=start+1
    right=end
    while left<=right:
        while left<=end and array[left]<=array[pivot]:
            left+=1
        while right>start and array[right]>=array[pivot]:
            right-=1
        if left>right:
            array[right],array[pivot]=array[pivot],array[right]
        else:
            array[left],array[right]=array[right],array[left]
    quick_sort(array,start,right-1)
    quick_sort(array,right+1,end)
quick_sort(array,0,len(array)-1)
print(array)
"""
# 계수정렬
"""
array=[7,5,9,0,3,1,6,2,9,1,4,8,0,5,2]

count=[0]*(max(array)+1)

for i in range(len(array)):
    count[array[i]]+=1

for i in range(len(count)):
    for j in range(count[i]):
        print(i, end=' ')
"""
# 성적이 낮은 순서로 학생 출력하기
"""
n=int(input())
array=[]
for i in range(n):
    data=input().split()
    array.append((data[0],data[1]))

array=sorted(array, key=lambda x: x[1])
for x in array:
    print(x[0], end=' ')
"""
# 이진탐색
"""
def binary(array, target, start, end):
    if start>end:
        return None
    mid=(start+end)//2
    if array[mid]==target:
        return mid
    elif array[mid]>target:
        return binary(array, target, start, mid-1)
    else:
        return binary(array, target, mid+1, end)
"""
# 부품 찾기
"""
n=int(input())
array=list(map(int,input().split()))
array.sort()
m=int(input())
x=list(map(int,input().split()))

def binary(array, target, start, end):
    while start<=end:
        mid=(start+end)//2
        if array[mid]==target:
            return mid
        elif array[mid]>target:
            end=mid-1
        else:
            start=mid+1
    return None

for i in x:
    result=binary(array,i,0,n-1)
    if result==None:
        print('no', end=' ')
    else:
        print('yes', end=' ')
"""
# 떡볶이 떡 만들기
"""
n, m = map(int,input().split())
array=list(map(int, input().split()))

start=0
end=max(array)


result=0
while start<=end:
    total=0
    mid=(start+end)//2
    for x in array:
        if x>mid:
            total+=(x-mid)
    if total<m:
        end=mid-1
    else:
        result=mid
        start=mid+1

print(result)
"""
# 1로 만들기
"""
n=int(input())
d=[0]*30001

for i in range(2,n+1):
    d[i]=d[i-1]+1
    if i%5==0:
        d[i]=min(d[i],d[i//5]+1)
    if i%3==0:
        d[i]=min(d[i],d[i//3]+1)
    if i%2==0:
        d[i]=min(d[i],d[i//2]+1)
print(d[n])
"""
# 개미 전사
"""
n=int(input())
array=list(map(int,input().split()))

d=[0]*101
d[0]=0
d[1]=max(array[0],array[1])

for i in range(2,n):
    d[i]=max(d[i-1], d[i-2]+array[i])

print(d[n-1])
"""
# 바닥 공사
"""
n=int(input())
d=[0]*1001

d[1]=1
d[2]=3
for i in range(3,n+1):
    d[i]=d[i-1]+d[i-2]*2
print(d[i]%796796)
"""
# 효율적인 화폐 구성
"""
n,m=map(int,input().split())
array=[]
for i in range(n):
    array.append(int(input()))
d=[10001]*(m+1)
d[0]=0
for i in range(n):
    for j in range(array[i],m+1):
        d[j]=min(d[j],d[j-array[i]]+1)

if d[m]==10001:
    print(-1)
else:
    print(d[m])
"""
# 미래 도시(플로이드 워셜 알고리즘)
"""
n,m=map(int, input().split())
INF=int(1e9)

graph=[[INF]*(n+1) for _ in range(n+1)]

for a in range(1,n+1):
    for b in range(1,n+1):
       if a==b:
            graph[a][b]=0
for i in range(m):
    a,b=map(int,input().split())
    graph[a][b]=1
    graph[b][a]=1

x,k=map(int,input().split())

for k in range(1,n+1):
    for a in range(1,n+1):
        for b in range(1,n+1):
            graph[a][b]=min(graph[a][b], graph[a][k]+graph[k][b])

distance=graph[1][k]+graph[k][x]

if distance>=INF:
    print("-1")
else:
    print(distance)
"""
# 전보(다익스트라 알고리즘)
"""
import heapq
import sys

input=sys.stdin.readline
INF=int(1e9)

n,m,c=map(int,input().split())
graph=[[] for i in range(n+1)]
distance=[INF]*(n+1)

for _ in range(m):
    x,y,z=map(int,input().split())
    graph[x].append((y,z))

def dijkstra(start):
    q=[]
    heapq.heappush(q,(0,start))
    distance[start]=0

    while q:
        dist,now=heapq.heappop(q)
        if distance[now]<dist:
            continue
        
        for i in graph[now]:
            cost=dist+i[1]
            
        if cost<distance[i[0]]:
            distance[i[0]]=cost
            heapq.heappush(q,(cost,i[0]))
dijkstra(c)

count=0
max_distance=0
for d in distance:
    if d!=INF:
        count+=1
        max_distance=max(max_distance,d)

print(count-1,max_distance)
"""
# 커리큘럼

data=list(map(int,input().split()))

print(data[1:-1])



















