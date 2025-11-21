from collections import deque

class Node:
    def __init__(self,val):
        self.value = val
        self.next = None

class linkedlist:
    def __init__(self):
        self.head = None
    
    def themCuoi(self, val):
        new_node = Node(val)
        if(self.head == None):
            self.head = new_node
            return
        cur = self.head
        while(cur.next != None):
            cur = cur.next
        cur.next = new_node
    
    def hienThi(self):
        p = self.head
        while(p !=None):
            print(f"{p.value}->",end='')
            p=p.next
            if(p== None):
                print(f"NULL")
    
    def layNode(self, val):
        p = self.head
        if(p == None):
            return
        while(p != None):
            if(p.value == val):
                return p.value
            p = p.next
        return None
    def toList(self):
        p = self.head
        ds = []
        while p!= None:
            ds.append(p.value)
            p =p.next
        return ds
#####
#sua danh sach ke truoc


def canh_to_matran(n, canh, vo_huong=True):
    ma_tran = [[0]*n for _ in range(n)]
    for edge in canh:
        if len(edge) == 2:
            u, v = edge
            w = 1  # mặc định vô trọng số
        else:
            u, v, w = edge
        ma_tran[u-1][v-1] = w
        if vo_huong:
            ma_tran[v-1][u-1] = w
    return ma_tran

def matran_to_dsk(n, ma_tran):
    dsk = [linkedlist() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if ma_tran[i][j] != 0:
                dsk[i].themCuoi(j+1)
    return dsk

def dsk_to_matran(n, dsk, vo_huong=True):
    ma_tran = [[0]*n for _ in range(n)]
    for i in range(n):
        for v in dsk[i].toList():
                ma_tran[i][v-1] = 1
                if vo_huong:
                    ma_tran[v-1][i] = 1
            
    return ma_tran

def matran_to_canh(n, ma_tran, vo_huong=True):
    canh = []
    for i in range(n):
        for j in range(n):
            if ma_tran[i][j] != 0:
                if vo_huong:
                    if i < j:  # tránh lặp lại trong vô hướng
                        canh.append((i+1,j+1))
                else:
                    canh.append((i+1,j+1))
    return canh

def dsk_to_canh(n, dsk, vo_huong=True):
    canh = []
    for i in range(n):
        for v in dsk[i].toList():
            if i+1 < v:
                canh.append((i+1,v))
            else:
                canh.append((i+1,v))
            
    return canh

def canh_to_dsk(n, canh, vo_huong=True):
    dsk = [linkedlist() for _ in range(n)]
    for edge in canh:
        if len(edge) == 2:
            u, v = edge
        else:
            u, v, w = edge
        dsk[u-1].themCuoi(v)
        if vo_huong:
            dsk[v-1].themCuoi(u)
    return dsk



#duyet do thi
def hasPathMTK_DFS(matran, u,v):  #ma tran ke
    n = len(matran)
    visited = [False]*n
    
    def path(cur):
        if(cur == v):
            return True
        visited[cur] = True
        for i in range(n):
            if(matran[cur][i] != 0 and not visited[i]):
                if(path(i)):
                    return True
        return False
    return path(u)

def hasPathMTK_BFS(matran, u,v):
    n = len(matran)
    visited = [False]*n
    queue = deque([u])
    visited[u] = True
    while queue:
        tam = queue.popleft()
        if(tam == v):
            return True
        for i in range(n):
            if(matran[tam][i] != 0 and visited[i] != True):
                visited[i] = True
                queue.append(i)
    return False

def hasPathDSK_DFS(ds, u, v):
    n = len(ds)
    visited = [False]*n

    def depth(cur):
        if(cur == v):
            return True
        visited[cur] = True
        current_node = ds[cur].head
        while current_node != None:
            neighbor = current_node.value
            if(visited[neighbor] != True):
                if(depth(neighbor)):
                    return True
        current_node = current_node.next
        return False
    return depth(u)

def hasPathDSK_BFS(ds, u ,v):
    n = len(ds)
    visited = [False]*n
    queue = deque([u])

    while queue:
        cur = queue.popleft()
        if cur == v:
            return True
        current_node = ds[cur].head
        while current_node != None:
            neighbor = current_node.value
            if visited[neighbor] != True:
                visited[neighbor] = True
                queue.append(cur)
            current_node = current_node.next
    return False


        

                




############################



def main():
    while True:
        print("\n===== CHUYỂN ĐỔI BIỂU DIỄN ĐỒ THỊ VÔ TRỌNG SỐ =====")
        print("1. Nhập danh sách cạnh")
        print("2. Nhập ma trận kề")
        print("3. Nhập danh sách kề")
        print("4. Thoát")
        choice = input("Chọn kiểu nhập: ")

        if choice == '4':
            print("Thoát chương trình.")
            break

        n = int(input("Nhập số đỉnh: "))
        vo_huong = input("Đồ thị vô hướng? (y/n): ") == 'y'

        if choice == '1':
            m = int(input("Nhập số cạnh: "))
            canh = []
            print("Nhập cạnh dạng: u v")
            for i in range(m):
                u, v = map(int, input(f"Cạnh {i+1}: ").split())
                canh.append((u,v))

            matran = canh_to_matran(n, canh, vo_huong)
            dsk = canh_to_dsk(n, canh, vo_huong)

        elif choice == '2':
            print("Nhập ma trận kề (0 là không có cạnh, 1 là có cạnh):")
            matran = []
            for i in range(n):
                row = list(map(int, input(f"Dòng {i+1}: ").split()))
                matran.append(row)

            canh = matran_to_canh(n, matran, vo_huong)
            dsk = matran_to_dsk(n, matran)

        elif choice == '3':
            dsk = [linkedlist() for _ in range(n)]
            m = int(input("Nhap so canh:"))
            print("Nhập danh sách kề: mỗi dòng là các đỉnh kề (vd: 2 5 6)")
            for _ in range(m):
                u,v = map(int,input("Nhap canh (u,v): ").split())
                dsk[u-1].themCuoi(v)
                if vo_huong:
                    dsk[v-1].themCuoi(u) 
                 

            matran = dsk_to_matran(n, dsk, vo_huong)
            canh = dsk_to_canh(n, dsk, vo_huong)
        
        k = input("ban co muon in do thi ra khong (y/n): " )
        
        if k== 'y':

            print("\n>>> Ma trận kề:")
            for row in matran:
                print(row)

            print("\n>>> Danh sách kề:")
            for i in dsk:
                i.hienThi()
                

            print("\n>>> Danh sách cạnh:")
            for edge in canh:
                print(*edge)

        print("\n-----------------------------------------\n")


main()
