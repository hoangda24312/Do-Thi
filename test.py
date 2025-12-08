from collections import deque
import heapq
import networkx as nx
import matplotlib.pyplot as plt


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

class DSU:
    def __init__(self,n):
        self.parent = list(range(n))
        self.rank = [0]*n
    
    def find(self,x):
        if(self.parent[x] != x):
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    #union bản nâng cấp có rank
    def union(self,x,y):
        xr,yr = self.find(x), self.find(y)
        if xr==yr: return False
        if(self.rank[xr] < self.rank[yr]):
            self.parent[xr] = yr
        elif (self.rank[yr] < self.rank[xr]):
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr]+=1
        return True
######
def veDSC(canh, n, vo_huong=True, check_trong_so=False):
    if vo_huong:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(1, n + 1))

    edge_labels = {}
    if check_trong_so:
        # Trường hợp CÓ TRỌNG SỐ: Cạnh là (u, v, w)
        G.add_weighted_edges_from(canh)
        edge_labels = {(u, v): w for u, v, w in canh}
    else:
        # Lọc lại danh sách cạnh để chỉ lấy 2 phần tử đầu tiên (u, v)
        canh_vo_trong_so = [(edge[0], edge[1]) for edge in canh]
        G.add_edges_from(canh_vo_trong_so) 

    # Cấu hình vị trí các nút
    pos = nx.spring_layout(G) 

    # Vẽ nút và cạnh
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=10)

    if check_trong_so and edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red') 
    
    plt.title("Trực quan hóa đồ thị")
    plt.show()

def draw_graph(n, edges, vo_huong=True, selected_edges=None, title="Trực quan hóa đồ thị"):
    """
    Vẽ đồ thị với khả năng tô màu các cạnh đã chọn.
    edges: danh sách cạnh ban đầu [(u, v, w), ...]
    selected_edges: danh sách các cạnh đã được chọn cho MST (để tô màu khác).
    """
    if vo_huong:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(1, n + 1))

    # Thêm tất cả các cạnh vào đồ thị
    G.add_weighted_edges_from(edges)

    # Lấy vị trí các nút
    pos = nx.spring_layout(G) 

    # Định nghĩa màu cho các cạnh
    all_edges = list(G.edges(data=True))
    edge_colors = ['gray'] * len(all_edges)
    edge_widths = [1] * len(all_edges)
    
    selected_edge_list = [(u, v) for u, v, w in (selected_edges or [])]
    
    # Tô màu các cạnh đã chọn
    for i, (u, v, data) in enumerate(all_edges):
        if (u, v) in selected_edge_list or (v, u) in selected_edge_list:
            edge_colors[i] = 'green'  # Cạnh đã chọn
            edge_widths[i] = 3
    
    # Lấy nhãn trọng số
    edge_labels = {(u, v): d['weight'] for u, v, d in all_edges if 'weight' in d}

    # Vẽ nút và cạnh
    plt.figure(figsize=(10, 7))
    nx.draw(
        G, pos, 
        with_labels=True, 
        node_color='skyblue', 
        node_size=1500, 
        font_size=12,
        edge_color=edge_colors,
        width=edge_widths
    )

    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red') 
    
    plt.title(title)
    plt.show(block=False) # Sử dụng block=False để cho phép nhiều hình vẽ
    plt.pause(4.5) 
    plt.close() 

def drawFlowGraph(n, capacity_edges, flow_data, path_edges=None, title="Trực quan hóa Luồng"):
    G = nx.DiGraph() # Luồng là đồ thị có hướng
    G.add_nodes_from(range(1, n + 1))

    # Xây dựng nhãn luồng và danh sách cạnh
    edge_labels = {}
    all_edges = []
    
    # Chuẩn hóa path_edges để dễ so sánh
    path_set = set((u, v) for u, v, w in (path_edges or []))

    for u, v, w in capacity_edges:
        # Tìm luồng ròng (net flow)
        flow = flow_data[u-1][v-1]
        
        # Đảm bảo luồng không vượt quá dung lượng
        flow_value = int(round(flow))
        if flow_value < 0:
            flow_value = 0
        
        G.add_edge(u, v, capacity=w, flow=flow_value, label=f"{flow_value}/{w}")
        edge_labels[(u, v)] = f"{flow_value}/{w}"
        all_edges.append((u, v))

    pos = nx.spring_layout(G) 

    # Định nghĩa màu và độ dày cho các cạnh
    edge_colors = []
    edge_widths = []
    
    for u, v in all_edges:
        is_path_edge = (u, v) in path_set
        
        if is_path_edge:
            edge_colors.append('red') # Tô màu đường tăng luồng
            edge_widths.append(3)
        else:
            edge_colors.append('gray')
            edge_widths.append(1.5)
            
    # Vẽ
    plt.figure(figsize=(10, 7))
    nx.draw(
        G, pos, 
        with_labels=True, 
        node_color='skyblue', 
        node_size=1500, 
        font_size=12,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True
    )

    if edge_labels:
        # Nhãn hiển thị Luồng/Dung lượng
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkgreen') 
    
    plt.title(title)
    plt.show(block=False) 
    plt.pause(4.5) 
    plt.close()

#kiểm tra đồ thị có phải 2 phía không
def checkBipartiesDSK(dsk, n):
    color = [-1]*n
    for i in range(n):
        if color[i] == -1:
            queue = deque([i])
            color[i] = 0
            while queue:
                u = queue.popleft()
                neighbor = dsk[u].toList()
                for v in neighbor:
                    v_idx = v-1
                    if v_idx < 0 or v_idx >= n:
                        print(f"Cảnh báo: Đỉnh {v} nằm ngoài phạm vi 1 đến {n}.")
                        continue
                    if(color[v_idx] == -1):
                        color[v_idx] = 1-color[u]
                        queue.append(v)
                    elif(color[v_idx] == color[u]):
                        return False
    return True

#thuật toán tìm đường đi ngắn nhất dijkstra
def Dijkstra(dsk, n,start):
    start_idx = start-1
    INF = float('inf')
    dist = [INF]*n
    prev = [None]*n

    dist[start_idx] = 0
    min_heap = [(0,start_idx)]
    while min_heap:
        d_u, u_idx = heapq.heappop(min_heap)
        if d_u > dist[u_idx]:
            continue
        for v,weight in dsk[u_idx].toListTrongSo():
            v_idx = v-1
            new_dist = dist[u_idx]+weight
            if new_dist < dist[v_idx]:
                dist[v_idx] = new_dist
                prev[v_idx] = u_idx
                heapq.heappush(min_heap,(new_dist,v_idx))
    return dist, prev




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
    else:
        for i in range(n):
            for v in dsk[i].toList():
                    ma_tran[i][v-1] = 1
                    if vo_huong:
                        ma_tran[v-1][i] = 1
    return ma_tran

def matran_to_canh(n, ma_tran,check_trong_so = True, vo_huong=True):
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
def hasPathMTK_DFS(matran, u,v):
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
#start là đỉnh bắt đầu, n là số đỉnh
def Prim(dsk, start, n): #code mẫu, thuần thuật toán không có trực quan
    visited = [False]*n
    min_heap = []
    cay_khung = []
    total_weight = 0
    egde_start = dsk[start-1].toListTrongSo()
    visited[start-1] = True
    #push start trước, và do danh sách lưu index 0 tới n-1 nên các đỉnh kề phải -1
    for neighbor_v, weight in egde_start:
        heapq.heappush(min_heap,(weight,start-1,neighbor_v-1))
    while min_heap != []:
        w,u,v = heapq.heappop(min_heap)
        if(visited[v]):
            continue
        #nhận cạnh u->v
        visited[v] = True
        cay_khung.append((u+1,v+1,w))
        total_weight+=weight
        #tiếp theo push các cạnh kề của đỉnh kề vào heap
        for x,w2 in dsk[v].toListTrongSo():
            x-=1
            if visited[x] != True:
                heapq.heappush(min_heap,(w2,v,x))
        if len(cay_khung) == n-1:
            break
    return cay_khung,total_weight
#Phần trực quan hóa
def Prim_visualize(dsk, start, n,vo_huong,check_trong_so):
    if not check_trong_so:
        print("Cần đồ thị có trọng số để chạy Prim.")
        return [], 0
    full_edges = dsk_to_canh(n, dsk, check_trong_so=True, vo_huong=vo_huong)
    
    visited = [False]*n
    min_heap = []
    cay_khung = []
    total_weight = 0
    start_idx = start - 1

    # Bắt đầu
    draw_graph(n, full_edges, vo_huong, cay_khung, title=f"Prim: Bắt đầu từ đỉnh {start}")
    
    # Bước 1: Khởi tạo
    egde_start = dsk[start_idx].toListTrongSo()
    visited[start_idx] = True
    for neighbor_v, weight in egde_start:
        heapq.heappush(min_heap, (weight, start_idx, neighbor_v - 1))
    
    print(f"Đã thăm đỉnh {start}. Đã thêm các cạnh kề vào Min-Heap.")
    draw_graph(n, full_edges, vo_huong, cay_khung, title=f"Prim: Đã thăm {start}")
    
    # Lặp lại đến khi cây khung hoàn thành
    while min_heap and len(cay_khung) < n - 1:
        w, u_idx, v_idx = heapq.heappop(min_heap)
        
        u_node, v_node = u_idx + 1, v_idx + 1
        print(f"\nXét cạnh: ({u_node}, {v_node}) với trọng số {w}")

        if visited[v_idx]:
            print(f"Đỉnh {v_node} đã được thăm. Bỏ qua.")
            continue
            
        # Nhận cạnh u->v
        visited[v_idx] = True
        cay_khung.append((u_node, v_node, w))
        total_weight += w
        
        print(f"--> CHỌN CẠNH: ({u_node}, {v_node}), Trọng số: {w}. Tổng trọng số: {total_weight}")
        draw_graph(n, full_edges, vo_huong, cay_khung, 
                   title=f"Prim: Đã chọn ({u_node}, {v_node}). Tổng W: {total_weight}")

        # Thêm các cạnh mới kề với đỉnh v_idx
        for x, w2 in dsk[v_idx].toListTrongSo():
            x_idx = x - 1
            if not visited[x_idx]:
                heapq.heappush(min_heap, (w2, v_idx, x_idx))
                
        print(f"Đã thêm các cạnh kề từ đỉnh {v_node} vào Min-Heap.")

    if len(cay_khung) == n - 1:
        print("\nHoàn thành Cây Khung Tối Thiểu (MST) bằng Prim.")
    else:
        print("\nKhông tìm thấy Cây Khung Tối Thiểu (MST). Đồ thị có thể không liên thông.")
        
    return cay_khung, total_weight


#7.2 thuật toán Kruskal
def Kruskal(dsk,n):
    edges = []
    for u in range(n):
        for v,w in dsk[u].toListTrongSo():
            if(v>u):
                edges.append((w,u,v-1))

    edges.sort()

    dsu = DSU(n)
    cay_khung = []
    total_weight = 0

    for w,u,v in edges:
        if(dsu.union(u,v)):
            cay_khung.append((u+1,v+1,w))
            total_weight+=w
        if(len(cay_khung) == n-1):
            break
    return cay_khung,total_weight 

#Phần trực quan hóa kruskal
def Kruskal_visualize(dsk, n,vo_huong,check_trong_so):
    if not check_trong_so:
        print("Cần đồ thị có trọng số để chạy Kruskal.")
        return [], 0
    
    edges = []
    for u in range(n):
        for v, w in dsk[u].toListTrongSo():
            if vo_huong and u + 1 < v:
                 edges.append((w, u, v - 1))
            elif not vo_huong: # Với đồ thị có hướng, thêm tất cả
                 edges.append((w, u, v - 1))
                 
    edges.sort()
    full_edges = dsk_to_canh(n, dsk, check_trong_so=True, vo_huong=vo_huong)
    
    dsu = DSU(n)
    cay_khung = []
    total_weight = 0

    # Bắt đầu
    draw_graph(n, full_edges, vo_huong, cay_khung, title="Kruskal: Sắp xếp cạnh và bắt đầu")
    
    print(f"Tổng số cạnh đã sắp xếp: {len(edges)}")
    
    # 2. Xử lý các cạnh đã sắp xếp
    for w, u_idx, v_idx in edges:
        u_node, v_node = u_idx + 1, v_idx + 1
        print(f"\nXét cạnh: ({u_node}, {v_node}) với trọng số {w}")
        
        # Kiểm tra xem cạnh có tạo thành chu trình không
        if dsu.find(u_idx) != dsu.find(v_idx):
            # Không tạo chu trình, CHỌN cạnh này
            dsu.union(u_idx, v_idx)
            cay_khung.append((u_node, v_node, w))
            total_weight += w
            print(f"--> CHỌN CẠNH: ({u_node}, {v_node}), Trọng số: {w}. Tổng trọng số: {total_weight}")
            
            draw_graph(n, full_edges, vo_huong, cay_khung, 
                       title=f"Kruskal: Đã chọn ({u_node}, {v_node}). Tổng W: {total_weight}")
            
            if len(cay_khung) == n - 1:
                break
        else:
            print(f"Cạnh ({u_node}, {v_node}) tạo thành chu trình. Bỏ qua.")

    if len(cay_khung) == n - 1:
        print("\nHoàn thành Cây Khung Tối Thiểu (MST) bằng Kruskal.")
    else:
        print("\nKhông tìm thấy Cây Khung Tối Thiểu (MST). Đồ thị có thể không liên thông.")

    return cay_khung, total_weight


#7.3 Ford Fulkersen

def bfsPath(do_thi, s,t,parent,n):
    visited = [False]*n
    queue = deque([s])
    visited[s] = True
    parent[s] = -1

    while queue:
        u = queue.popleft()
        for i in range(n):
            if(do_thi[u][i] >0 and visited[i] != True):
                queue.append(i)
                visited[i] = True
                parent[i] = u
    
    return visited[t] #nếu đến được t thì có đồ thị tăng luồng
            



def fordFulkersen(matran,n,s_node, t_node):
    s = s_node - 1
    t = t_node - 1
    thang_du = matran
    max_flow = 0 #luồng cực đại
    parent = [0]*n #mảng lưu đường đi
    while(bfsPath(thang_du,s,t,parent,n)):
        path_flow = float('inf')
        v = t
        while v !=s: #duyệt ngược từ t về s
            u = parent[v]
            path_flow = min(path_flow, thang_du[u][v])
            v = u
        max_flow+=path_flow
        v = t
        while v != s:
            u = parent[v]
            thang_du[u][v] -= path_flow
            thang_du[v][u] += path_flow
            v= u
    return max_flow


def fordFulkersen_visualize(matran_goc, n, s_node, t_node):
    s = s_node - 1
    t = t_node - 1

    thang_du = [row[:] for row in matran_goc] 
    
    luong_hien_tai = [[0] * n for _ in range(n)]
    
    max_flow = 0
    parent = [0] * n 
    
    # Lấy danh sách cạnh ban đầu để vẽ nền
    full_edges = matran_to_canh(n, matran_goc, check_trong_so, vo_huong=False) 

    print("--- BẮT ĐẦU FORD-FULKERSON (EDMONDS-KARP) ---")
    drawFlowGraph(n, full_edges, luong_hien_tai, title=f"Luồng ban đầu (F=0)")

    # Vòng lặp chính
    vong_lap = 1
    while(bfsPath(thang_du, s, t, parent, n)):
        path_flow = float('inf')
    
        v = t
        path_edges = [] # Lưu các cạnh trên đường tăng luồng để trực quan hóa
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, thang_du[u][v])
            path_edges.append((u + 1, v + 1, matran_goc[u][v])) 
            v = u
        path_edges.reverse() 

        max_flow += path_flow
        
        v = t
        while v != s:
            u = parent[v]
        
            thang_du[u][v] -= path_flow
            thang_du[v][u] += path_flow
            if matran_goc[u][v] > 0:
                luong_hien_tai[u][v] += path_flow
            elif matran_goc[v][u] > 0:
                luong_hien_tai[v][u] -= path_flow
                
            v = u
            
        print(f"\n--- Vòng {vong_lap} ---")
        print(f"Đường tăng luồng: {path_edges} (W: {path_flow})")
        print(f"Luồng cực đại mới: F = {max_flow}")
        # 5. Trực quan hóa
        # Gọi hàm vẽ. Giả định draw_graph có thể chấp nhận nhãn là chuỗi
        drawFlowGraph(n, full_edges, luong_hien_tai,path_edges, title=f"Vòng {vong_lap}: F={max_flow}")

        vong_lap += 1
        
    print(f"\n--- KẾT THÚC ---")
    print(f"Luồng Cực Đại là: {max_flow}")
    return max_flow

#7.3 Thuật toán Fleury
def isEuler(dsk, n):
    node = []
    for i in range(n):
        bac = len(dsk[i].toList())
        if bac % 2 != 0:
            node.append(i+1)
    start_node = None
    if len(node) == 0:
        print("Có chu trình euler")
        start_node = 1
    elif len(node) <=2:
        print("Có đường đi Euler, không có chu trình")
        start_node = node[0]
    else:
        print(f"Đồ thị có {len(node)} bậc lẻ, không có chu trình/đường đi euler")
        return None, None
    return start_node, len(node)

def is_bridge(dsk_temp, u,v,vo_huong):
    if(len(dsk[u].toList()) == 1 or len(dsk[v].toList()) == 1):
        return False
    matran_temp = dsk_to_matran(n,dsk_temp,check_trong_so=False,vo_huong=vo_huong)
    matran_temp[u][v] = 0
    if vo_huong:
        matran_temp[v][u] = 0
    still_connected = hasPathDSK_DFS(matran_temp,u,v)
    return not still_connected

def Fleury(dsk,n,vo_huong = True):
    start_node, num = isEuler(dsk,n)
    if start_node == None:
        return []
    #sao chép dsk qua temp để không xóa cạnh dsk
    dsk_temp = [linkedlist() for _ in range(n)]
    for i in range(n):
        for val, w in dsk[i].toListTrongSo():
            dsk_temp[i].themCuoi(val,w)
    euler_path = [start_node]
    current = start_node
    while True:
        current_idx = current-1
        possible_neighbor = dsk_temp[current_idx].toList()
        if not possible_neighbor: #dừng khi không còn cạnh kề
            break
        # Tìm cạnh (current_node, v) để chọn
        for neightbor in possible_neighbor:
            #Quy tắc 1: nếu chỉ còn 1 cạnh duy nhất thì phải chọn nó
            if(len(possible_neighbor) == 1):
                selected_neighbor = neightbor
                break
            #Quy tắc 2: không chọn cầu trừ khi nó là duy nhất
            if not is_bridge(dsk_temp,current_idx,neightbor,vo_huong):
                selected_neighbor = neightbor
                break
        # Nếu đã duyệt qua hết mà không tìm được cạnh không phải cầu, thì phải chọn cầu (cạnh còn lại)
        if selected_neighbor is None:
            selected_neighbor = possible_neighbor[0]
        next_node = selected_neighbor
        dsk_temp[current_idx].xoaNode(next_node)
        if vo_huong:
            dsk_temp[next_node-1].xoaNode(current)
        euler_path.append(next_node)
        current = next_node
    print("\n--- THUẬT TOÁN FLEURY ---")
    print(f"Chu trình/Đường đi Euler: {euler_path}")
    return euler_path


        






dsk = []
matran = []
canh = []
n = 0
vo_huong = True
check_trong_so = False



############################

#Cho nhập ma trận chưa có trọng số trước->thêm dần tìm kiếm->chỉnh sửa lại cho trọng số->


#thay đổi 1: Thêm trọng số vào class danh sách liên kết và sửa các hàm để nhận trọng số trong vài trường hợp
def main():
    global dsk, matran,canh,n,vo_huong,check_trong_so
    while True:
        print("\n===== BIỂU DIỄN ĐỒ THỊ TRỌNG SỐ =====")
        #dồn các danh sách vào 1 lựa chọn và cho người dùng chọn
        print("1.Vẽ đồ thị")
        print("2. Nhập danh sách")
        print("3.Tìm đường đi ngắn nhất")
        print("4.Duyệt đồ thị theo các chiến lược: BFS & DFS")
        print("5.Kiểm tra đồ thị 2 phía")
        print("6.Chuyển đổi qua lại giữa các đồ thị")
        print("7.Trực quan hóa các thuật toán")
        print("10. Thoát")
        choice = input("Chọn kiểu nhập: ")

        if choice == '10':
            print("Thoát chương trình.")
            break

        if choice == '1':
            if canh == []:
                print("Chưa có danh sách cạnh, hãy chuyển đổi hoặc nhập danh sách cạnh")
            else:
                veDSC(canh,n,vo_huong,check_trong_so)

        if choice == '2':
            print("1. Nhập danh sách cạnh")
            print("2. Nhập ma trận kề")
            print("3. Nhập danh sách kề")
            choice_DanhSach = input("bạn muốn dùng danh sách nào ?")
            if choice_DanhSach =='1':
                n = int(input("Nhập số đỉnh: "))
                vo_huong = input("Đồ thị vô hướng? (y/n): ") == 'y'
                check = input("Ma trận có trọng số không ? y/n")
                m = int(input("Nhập số cạnh: "))
                canh = []
                if check == 'y':
                    print("Nhập cạnh dạng: u v w")
                    for i in range(m):
                        u, v,w = map(int, input(f"Cạnh {i+1}: ").split())
                        canh.append((u,v,w))
                    check_trong_so = True
                else:
                    print("Nhập cạnh dạng: u v")
                    for i in range(m):
                        u, v = map(int, input(f"Cạnh {i+1}: ").split())
                        canh.append((u,v))

            elif choice_DanhSach == '2':
                n = int(input("Nhập số đỉnh: "))
                vo_huong = input("Đồ thị vô hướng? (y/n): ") == 'y'
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
        
            elif choice_DanhSach == '3':
                n = int(input("Nhập số đỉnh: "))
                vo_huong = input("Đồ thị vô hướng? (y/n): ") == 'y'
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

        elif choice == '3':
            if dsk == []:
                print("Bạn chưa có danh sách kề, vui lòng chuyển đổi hoặc nhập")
            else:
                start = int(input("Nhập đỉnh bắt đầu"))
                dist, prev = Dijkstra(dsk,n,start)
                print(f"Đường đi ngắn nhất là",dist)
                print(f"Đỉnh trước:",prev)


        elif choice == '4':
            print("1.Ma trận kề")
            print("2.Danh sách kề")
            find_choice = input("bạn muốn chọn tìm kiếm trên ma trận nào?")
            u = int(input("Chọn đỉnh bắt đầu"))
            v = int(input("chọn đỉnh kết thúc"))
            type_find = int(input("Bạn muốn tìm kiếm theo chiều sâu hay rộng(0 là chiều sâu, 1 là chiều rộng)"))
            if find_choice == '1':
                if matran == []:
                    print("chưa có ma trận này")
                else:
                    if(type_find == 0):
                        if(hasPathMTK_DFS(matran,u-1,v-1)):
                            print(f"Có đường đi từ đỉnh {u} tới đỉnh {v}")
                        else:
                            print(f"không có đường đi từ đỉnh {u} tới đỉnh {v}")
                    else:
                        if(hasPathMTK_BFS(matran,u-1,v-1)):
                            print(f"Có đường đi từ đỉnh {u} tới đỉnh {v}")
                        else:
                            print(f"không có đường đi từ đỉnh {u} tới đỉnh {v}")
            elif find_choice == '2':
                if dsk == []:
                    print("chưa có danh sách kề")
                else:
                    if(type_find == 0):
                        if(hasPathDSK_DFS(dsk,u-1,v-1)):
                            print(f"Có đường đi từ đỉnh {u} tới đỉnh {v}")
                        else:
                            print(f"không có đường đi từ đỉnh {u} tới đỉnh {v}")
                    else:
                        if(hasPathDSK_BFS(dsk,u-1,v-1)):
                            print(f"Có đường đi từ đỉnh {u} tới đỉnh {v}")
                        else:
                            print(f"không có đường đi từ đỉnh {u} tới đỉnh {v}")



        elif choice == '5':
            if dsk == []:
                print("Chưa có danh sách kề, vui lòng chuyển đổi hoặc nhập danh sách kề")
            else:
                tam = checkBipartiesDSK(dsk,n)
                if(tam == True):
                    print("Đồ thị là đồ thị 2 phía")
                else:
                    print("Đồ thị không phải là đồ thị 2 phía")



        elif choice == '6':
            print("1.danh sách cạnh->ma trận kề")
            print("2.danh sách cạnh->danh sách kề")
            print("3.ma trận kề->danh sách cạnh")
            print("4.ma trận kề->danh sách kề")
            print("5.danh sách kề->ma trận kề")
            print("6.danh sách kề->danh sách cạnh")
            chon = input("Bạn muốn đổi đồ thị nào sang nào ?")
            if chon == '1':
                if canh == []:
                    print("chưa có danh sách cạnh, chọn đồ thị khác hoặc nhập danh sách cạnh")
                else:
                    matran = canh_to_matran(n, canh, vo_huong)
                    print("đã chuyển danh sách cạnh thành ma trận kề thành công")
            elif chon == '2':
                if canh == []:
                    print("chưa có danh sách cạnh, chọn đồ thị khác hoặc nhập danh sách cạnh")
                else:
                    dsk = canh_to_dsk(n, canh, vo_huong)
                    print("đã chuyển danh sách cạnh thành danh sách kề thành công")
            elif chon == '3':
                if matran == []:
                    print("chưa có ma trận kề, chọn đồ thị khác hoặc nhập danh sách cạnh")
                else:
                    canh = matran_to_canh(n, matran,check_trong_so, vo_huong)
                    print("đã chuyển ma trận kề thành danh sách cạnh thành công")
            elif chon == '4':
                if matran == []:
                    print("chưa có ma trận kề, chọn đồ thị khác hoặc nhập danh sách cạnh")
                else:
                    dsk = matran_to_dsk(n, matran,check_trong_so)
                    print("đã chuyển ma trận kề thành danh sách kề thành công")
            elif chon == '5':
                if dsk == []:
                    print("chưa có danh sách kề, chọn đồ thị khác hoặc nhập danh sách cạnh")
                else:
                    matran = dsk_to_matran(n, dsk,check_trong_so, vo_huong)
                    print("đã chuyển danh sách kề thành ma trận kề thành công")
            elif chon == '6':
                if dsk == []:
                    print("chưa có danh sách kề, chọn đồ thị khác hoặc nhập danh sách cạnh")
                else:
                    canh = dsk_to_canh(n, dsk,check_trong_so, vo_huong)
                    print("đã chuyển danh sách kề thành danh sách cạnh thành công")
        elif choice == '7':
            print("1 Prim")
            print("2 Kruskal")
            print("3 Ford fulkerson")
            print("4 Fleury")
            print("5 Hierholzer")
            thuat_toan = input("Bạn muốn chọn thuật toán nào")
            if thuat_toan == '1':
                if dsk == []:
                    print("Chưa có Danh sách kề. Vui lòng nhập hoặc chuyển đổi sang Danh sách kề có trọng số.")
                elif not check_trong_so:
                    print("Thuật toán Prim và Kruskal yêu cầu đồ thị có trọng số. Vui lòng nhập lại đồ thị.")
                else:
                    start_node = int(input("Nhập đỉnh bắt đầu cho Prim (1 đến n): "))
                    if 1 <= start_node <= n:
                        mst, total_weight = Prim_visualize(dsk, start_node, n,vo_huong,check_trong_so)
                        print(f"\nCây Khung Tối Thiểu (Prim): {mst}, Tổng trọng số: {total_weight}")
                    else:
                        print("Đỉnh bắt đầu không hợp lệ.")
            elif thuat_toan == '2':
                if dsk == []:
                    print("Chưa có Danh sách kề. Vui lòng nhập hoặc chuyển đổi sang Danh sách kề có trọng số.")
                elif not check_trong_so:
                    print("Thuật toán Prim và Kruskal yêu cầu đồ thị có trọng số. Vui lòng nhập lại đồ thị.")
                else:
                    mst, total_weight = Kruskal_visualize(dsk,n,vo_huong,check_trong_so)
                    print(f"\nCây Khung Tối Thiểu (Kruskal): {mst}, Tổng trọng số: {total_weight}")
            elif thuat_toan =='3':
                if matran == []:
                    print("Chưa có ma trận kề, hãy nhập hoặc chuyển đổi ma trận kề")
                elif check_trong_so == False:
                    print("Đồ thị cần có trọng số,vui lòng nhập lại")
                else:
                    s,t = map(int,input("Nhập đỉnh bắt đầu và kết thúc (s và t): ").split())
                    fordFulkersen_visualize(matran,n,s,t)
        print("\n-----------------------------------------\n")


main()
