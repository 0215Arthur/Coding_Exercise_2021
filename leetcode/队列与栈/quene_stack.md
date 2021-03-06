
- [队列](#队列)
  - [基础知识](#基础知识)
    - [简单实现](#简单实现)
    - [标准库： Queue](#标准库-queue)
    - [基于队列进行广度优先搜索BFS](#基于队列进行广度优先搜索bfs)
      - [常用模板1（queue实现BFS搜索最短路径）:](#常用模板1queue实现bfs搜索最短路径)
      - [常用模板2 (queue+hash set实现BFS/不重复访问)](#常用模板2-queuehash-set实现bfs不重复访问)
  - [622. 设计循环队列 [Medium]](#622-设计循环队列-medium)
  - [200. 岛屿数量 [Medium]](#200-岛屿数量-medium)
  - [752. 打开转盘锁 [Medium]](#752-打开转盘锁-medium)
## 队列
### 基础知识
- 基本特性：
  - **queue**: 先入先出
  - 入队 enqueue： 队尾插入
  - 出队 denqueue: 移除队首
#### 简单实现
```#include <iostream>
class MyQueue {
    private:
        // store elements
        vector<int> data;       
        // a pointer to indicate the start position
        int p_start;            
    public:
        MyQueue() {p_start = 0;}
        /** Insert an element into the queue. Return true if the operation is successful. */
        bool enQueue(int x) {
            data.push_back(x);
            return true;
        }
        /** Delete an element from the queue. Return true if the operation is successful. */
        bool deQueue() {
            if (isEmpty()) {
                return false;
            }
            p_start++;
            return true;
        };
        /** Get the front item from the queue. */
        int Front() {
            return data[p_start];
        };
        /** Checks whether the queue is empty or not. */
        bool isEmpty()  {
            return p_start >= data.size();
        }
};
```
#### 标准库： Queue
- queue<int> p;
- p.push(x)
- p.pop();
- p.front();
- p.empty()

#### 基于队列进行广度优先搜索BFS
BFS：
  - 类似于树的层次遍历，优先访问同一层的节点，用队列存储当前层节点，并将遍历到的节点的子节点添加到队列中
##### 常用模板1（queue实现BFS搜索最短路径）:
```
int BFS(Node root, Node target) {
    Queue<Node> q; // store all nodes which are waiting to be processed
    q.push(root); // initialize
    int step = 0;
    while(!q.empty()) {
        s = q.size();
        for (int i = 0; i < s; i++) {
            Node cur = q.pop();
            if (cur == target) {
                return step;
            }
            for (Node next : neighbor(cur)) {
                q.push(next);
            }
        }
        step ++;
    }
    return -1;
}
```

##### 常用模板2 (queue+hash set实现BFS/不重复访问)

- 某些情况下，我们需要避免遍历时重复访问节点，可能会出现loop
- 可以利用hash表记录以访问过的节点
```/**
 * Return the length of the shortest path between root and target node.
 */
int BFS(Node root, Node target) {
    Queue<Node> queue;  // store all nodes which are waiting to be processed
    Set<Node> used;     // store all the used nodes
    int step = 0;       // number of steps neeeded from root to current node
    // initialize
    add root to queue;
    add root to used;
    // BFS
    while (queue is not empty) {
        step = step + 1;
        // iterate the nodes which are already in the queue
        int size = queue.size();
        for (int i = 0; i < size; ++i) {
            Node cur = the first node in queue;
            return step if cur is target;
            for (Node next : the neighbors of cur) {
                if (next is not in used) {
                    add next to queue;
                    add next to used;
                }
            }
            remove the first node from queue;
        }
    }
    return -1;          // there is no path from root to target
}
```


### 622. 设计循环队列 [Medium]
- 利用数组进行队列数据管理，出队操作可以从原来的O(N)降至O(1)
- 参考代码中设计队列容量为 `k+1`
  - 主要是为了留出一个位置，用来判断队列是否满；
  - rear更严格的定义为：下一个元素插入的位置，但目前并未插入
  - 当`(rear+1)==k+1`时，数组实际存储了k个元素，当`(rear+1)%(k+1)==front`时说明数组已满，这种写法避免了最开始front和rear都为0时的特殊情况
- 解析参考： https://zhuanlan.zhihu.com/p/79163010
```class MyCircularQueue {
private:
    vector<int> data;
    int capacity;
    int front = 0;
    int rear = 0;
public:
    MyCircularQueue(int k) {
        capacity = k + 1;
        data.assign(capacity, 0);
    }
    
    bool enQueue(int value) {
        if (isFull())
            return false;
        data[rear] = value;
        rear = (rear + 1) % capacity;
        return true;
    }
    
    bool deQueue() {
        if (isEmpty())
            return false;
        
        front = (front + 1) % capacity;
        return true;

    }
    
    int Front() {
        return isEmpty() ? -1 : data[front];
    }
    
    int Rear() {
        return isEmpty() ? -1 : data[(rear - 1 + capacity) % capacity];

    }
    
    bool isEmpty() {
        if (front == rear) {
            return true;
        }
        return false;
    }
    
    bool isFull() {
        if (front == (rear + 1) % capacity)
            return true;
        return false;
    }
};
```


### 200. 岛屿数量 [Medium]
- **基于BFS遍历进行实现**
  - 从元素1的地方开始BFS遍历，每次BFS遍历就把1更新为0
  - BFS遍历中对每个位置添加其周围4个方向的1元素进入队列 （若没有进行完全4个方向的遍历，部分情况可能遍历不完整）
  - **BFS的遍历次数即为最终结果**
- 时间复杂度： O(MN) 每个元素都要访问一次
- 空间复杂度： O(min(M,N)) 最大队列长度为min(M,N) 

```class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        if (grid.empty())
            return 0;
        int res = 0;
        int m = grid.size();
        int n = grid[0].size();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '0')
                    continue;
                res ++;
                queue<pair<int,int>> q;
                q.push({i,j});
                while (!q.empty()) {
                    int s = q.size();
                    for (int k = 0; k < s; k++) {
                        pair<int,int> cur = q.front();
                        q.pop();
                        int row = cur.first;
                        int col = cur.second;
                        if (row - 1 >= 0 && grid[row - 1][col] == '1') {
                            q.push({row - 1, col});
                            grid[row - 1][col] = '0';
                        }
                        if (row + 1 < m && grid[row + 1][col] == '1') {
                            q.push({row + 1, col});
                            grid[row + 1][col] = '0';
                        }
                        if (col - 1 >= 0 && grid[row][col - 1] == '1') {
                            q.push({row, col - 1});
                            grid[row][col - 1] = '0';
                        }
                        if (col + 1 < n && grid[row][col + 1] == '1') {
                            q.push({row, col + 1});
                            grid[row][col + 1] = '0';
                        }
                    }
                }
            }
        }
        return res;
    }
};
```

### 752. 打开转盘锁 [Medium]
- 加了访问限制的遍历问题，求最小步数
- 使用BFS遍历解题，需要利用哈希表存储限制节点和已访问节点
- 时间复杂度： O($N^2*A^N+D$), 空间复杂度： O($A^N+D$) A为10 数字数量；N为状态位数；D为限制状态的数量

```class Solution {
public:
    bool find_dead(vector<string>& deadends, string target) {
        vector<string>::iterator iter=find(deadends.begin(),deadends.end(),"0000");
        if (iter == deadends.end())
            return false;
        return true;
    }
    int openLock(vector<string>& deadends, string target) {
        int res = 0;
        queue<string> q;
        unordered_set<string> visited; // 使用哈希表存储访问过的节点/dead节点，避免重复访问
        visited.insert(deadends.begin(),deadends.end());
        if (find_dead(deadends,"0000"))
            return -1;
        q.push("0000");
        while (!q.empty()) {
            int s = q.size();
            for (int i = 0; i < s; i++) {
                string cur = q.front();
                q.pop();
                if (cur == target) 
                    return res;
                for (int j = 0; j < 4; j++) {
                    for (int k = -1; k < 2; k+=2) {
                        string tmp = cur;
                        
                        tmp[j] = char((cur[j] - '0' + 10 + k) % 10 + '0');
                        //cout << cur << ": "<< tmp <<endl;
                        if (!visited.count(tmp)) {
                            q.push(tmp);
                            visited.insert(tmp);
                        }
                    }
                }
            }
            res ++; 
        }
        return -1;

    }
};
```