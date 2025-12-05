from collections import deque
import heapq

class Node:
    def __init__(self,val,W = 0):
        self.value = val
        self.trong_so = W
        self.next = None

class linkedlist:
    def __init__(self):
        self.head = None
    
    def themCuoi(self, val,W=0):
        new_node = Node(val,W)
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
    
    def toListTrongSo(self):
        p = self.head
        ds = []
        while p!= None:
            ds.append((p.value,p.trong_so))
            p =p.next
        return ds
    
    def xoaNode(self, val):
        if self.head == None:
            return
        while self.head != None and self.head == val:
            self.head = self.head.next
        p = self.head
        while(p!=None and p.next != None):
            if(p.next.value == val):
                q= p.next
                p.next = q.next
                del q
            else:
                p = p.next
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

def matran_to_dsk(n, ma_tran, check_trong_so=False):
    dsk = [linkedlist() for _ in range(n)]
    if check_trong_so == True:
        for i in range(n):
            for j in range(n):
                if ma_tran[i][j] != 0:
                    dsk[i].themCuoi(j+1,ma_tran[i][j])
    else:
        for i in range(n):
            for j in range(n):
                if ma_tran[i][j] != 0:
                    dsk[i].themCuoi(j+1)
    return dsk

def dsk_to_matran(n, dsk,check_trong_so=False, vo_huong=True):
    ma_tran = [[0]*n for _ in range(n)]
    if check_trong_so == True:
        for i in range(n):
            for v,w in dsk[i].toListTrongSo():
                    ma_tran[i][v-1] = w
                    if vo_huong:
                        ma_tran[v-1][i] = w

    for i in range(n):
        for v in dsk[i].toList():
                ma_tran[i][v-1] = 1
                if vo_huong:
                    ma_tran[v-1][i] = 1
    return ma_tran

def matran_to_canh(n, ma_tran,check_trong_so=False, vo_huong=True):
    canh = []
    if check_trong_so==True:
        for i in range(n):
            for j in range(n):
                if ma_tran[i][j] != 0:
                    if vo_huong:
                        if i < j:  # tránh lặp lại trong vô hướng
                            canh.append((i+1,j+1,ma_tran[i][j]))
                    else:
                        canh.append((i+1,j+1,ma_tran[i][j]))
    else:
        for i in range(n):
            for j in range(n):
                if ma_tran[i][j] != 0:
                    if vo_huong:
                        if i < j:  # tránh lặp lại trong vô hướng
                            canh.append((i+1,j+1))
                    else:
                        canh.append((i+1,j+1))
    return canh

def dsk_to_canh(n, dsk,check_trong_so = False, vo_huong=True):
    canh = []
    if check_trong_so == True:
        for i in range(n):
            for v,w in dsk[i].toListTrongSo():
                if i+1 < v:
                    canh.append((i+1,v,w))
                else:
                    canh.append((i+1,v,w))
    else:
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
        dsk[u-1].themCuoi(v,w)
        if vo_huong:
            dsk[v-1].themCuoi(u,w)
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

#############################
#chu trinh euler va hamilton

def chuTrinhEuler(dsk,u, vo_huong = True): #u la dinh bat ki
    stack = deque([u])
    EC = [] #chu trinh euler
    while stack:
        top = stack[-1]  #lay dinh tren cung
        ds = dsk[top-1].toList()
        if len(ds) > 0:
            neighbor = ds[0] #canh dau tien
            stack.append(neighbor)
            dsk[top-1].xoaNode(neighbor)
            if vo_huong:
                dsk[neighbor-1].xoaNode(top)
        else:
            top = stack.pop()
            EC.append(top)
    return EC

def quayLui(matran, path, visited,k):
    for i in range(len(matran)):
        if(matran[path[k-1]][i] == 1 and visited[i] != True):
            path[k] = i
            visited[i] = True
            if(k == len(matran)-1):
                if(matran[path[k]][path[0]] == 1):
                    chu_trinh = path + [path[0]]
                    print("chu trinh hamilton tim duoc:", chu_trinh)
            else:
                quayLui(matran,path,visited,k+1)
            visited[i] = False
            path[k] = 0




def chuTrinhHamilton(matran,u,vo_huong = True):
    n = len(matran)
    path = [0 for _ in range(n)]
    visited = [False for _ in range(n)]
    path[0] = u
    visited[u] = True
    quayLui(matran,path,visited,1)

##################################
#cac thuat toan nang cao
#kruskal thì chọn những cái nhỏ nhất rồi bỏ vào, không cần thiết phải kề với đỉnh trong cây, miễn là nó không tạo thành chu trình
#prim thì chọn cái nhỏ nhất rồi mở rộng vùng từ từ, nói cách khác các cạnh có trọng số nhỏ nhất phải kề với đỉnh đã có trong cây
#7.1 Thuat toan Prim
"""
Bắt đầu từ một đỉnh bất kỳ
Đưa toàn bộ cạnh kề vào heap

Mỗi lần lấy ra cạnh nhỏ nhất

Nếu nối vào một đỉnh mới → nhận

Thêm các cạnh từ đỉnh mới vào heap

Lặp lại đến khi có đủ n-1 cạnh
"""
def Prim(dsk):







############################

#Cho nhập ma trận chưa có trọng số trước->thêm dần tìm kiếm->chỉnh sửa lại cho trọng số->


#thay đổi 1: Thêm trọng số vào class danh sách liên kết và sửa các hàm để nhận trọng số trong vài trường hợp
def main():
    while True:
        print("\n===== CHUYỂN ĐỔI BIỂU DIỄN ĐỒ THỊ VÔ TRỌNG SỐ =====")
        #dồn các danh sách vào 1 lựa chọn và cho người dùng chọn
        print("1. Nhập danh sách")
        print("4. In ra chu trình euler")
        print("5.Chu trinh hamilton")
        print("6. Thoát")
        choice = input("Chọn kiểu nhập: ")

        if choice == '6':
            print("Thoát chương trình.")
            break

        n = int(input("Nhập số đỉnh: "))
        vo_huong = input("Đồ thị vô hướng? (y/n): ") == 'y'

        if choice == '1':
            print("1. Nhập danh sách cạnh")
            print("2. Nhập ma trận kề")
            print("3. Nhập danh sách kề")
            choice_DanhSach = input("bạn muốn dùng danh sách nào ?")
            if choice_DanhSach =='1':
                check = input("Ma trận có trọng số không ? y/n")
                m = int(input("Nhập số cạnh: "))
                canh = []
                if check == 'y':
                    print("Nhập cạnh dạng: u v w")
                    for i in range(m):
                        u, v,w = map(int, input(f"Cạnh {i+1}: ").split())
                        canh.append((u,v,w))
                else:
                    print("Nhập cạnh dạng: u v")
                    for i in range(m):
                        u, v = map(int, input(f"Cạnh {i+1}: ").split())
                        canh.append((u,v))

                matran = canh_to_matran(n, canh, vo_huong)
                dsk = canh_to_dsk(n, canh, vo_huong)

            elif choice_DanhSach == '2':
                print("Nhập ma trận kề (0 là không có cạnh, 1 là có cạnh):")
                check = input("Ma trận có trọng số không ? y/n")
                matran = []
                if check == 'y':
                    check_trong_so = True
                else:
                    check_trong_so = False
                for i in range(n):
                    row = list(map(int, input(f"Dòng {i+1}: ").split()))
                    matran.append(row)

                canh = matran_to_canh(n, matran,check_trong_so, vo_huong)
                dsk = matran_to_dsk(n, matran,check_trong_so)

            elif choice_DanhSach == '3':
                dsk = [linkedlist() for _ in range(n)]
                check = input("Ma trận có trọng số không ? y/n")
                m = int(input("Nhap so canh:"))
                print("Nhập danh sách kề: mỗi dòng là các đỉnh kề (vd: 2 5, 2 4)")
                if check == 'y':
                    for _ in range(m):
                        u,v,w = map(int,input("Nhap canh (u,v,w): ").split())
                        dsk[u-1].themCuoi(v,w)
                        if vo_huong:
                            dsk[v-1].themCuoi(u,w) 
                    check_trong_so = True
                else:
                    for _ in range(m):
                        u,v = map(int,input("Nhap canh (u,v): ").split())
                        dsk[u-1].themCuoi(v)
                        if vo_huong:
                            dsk[v-1].themCuoi(u) 
                    check_trong_so = False
                    

                matran = dsk_to_matran(n, dsk,check_trong_so, vo_huong)
                canh = dsk_to_canh(n, dsk,check_trong_so, vo_huong)
        
        elif choice == '4':
            if(dsk == None):
                print("Ban chua nhap danh sach ke")
            else:
                start = int(input("Nhap dinh bat dau:"))
                print(chuTrinhEuler(dsk,start,vo_huong))
        
        elif choice == '5':
            if(matran == None):
                print("ban chua nhap ma tran")
            else:
                start = int(input("Nhap dinh bat dau u-1 (neu la dinh 2 thi nhap 1):"))
                chuTrinhHamilton(matran,start,vo_huong)
            
        
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
