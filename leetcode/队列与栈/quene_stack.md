
- [队列](#队列)
  - [基础知识](#基础知识)
    - [简单实现](#简单实现)
    - [标准库： Queue](#标准库-queue)
    - [基于队列进行广度优先搜索BFS](#基于队列进行广度优先搜索bfs)
      - [常用模板1（queue实现BFS搜索最短路径）:](#常用模板1queue实现bfs搜索最短路径)
      - [常用模板2 (queue+hash set实现BFS/不重复访问)](#常用模板2-queuehash-set实现bfs不重复访问)
  - [622. 设计循环队列 [Medium]*](#622-设计循环队列-medium)
  - [200. 岛屿数量 [Medium]](#200-岛屿数量-medium)
  - [695. 岛屿的最大面积](#695-岛屿的最大面积)
  - [1254. 统计封闭岛屿的数目](#1254-统计封闭岛屿的数目)
  - [752. 打开转盘锁 [Medium]](#752-打开转盘锁-medium)
  - [279. 完全平方数 [Medium]](#279-完全平方数-medium)
    - [BFS套模板](#bfs套模板)
    - [dp解法](#dp解法)
  - [队列题目小结](#队列题目小结)
- [栈 stack](#栈-stack)
  - [基础知识](#基础知识-1)
  - [基本实现](#基本实现)
  - [主要API](#主要api)
  - [155. 最小栈 [Easy]](#155-最小栈-easy)
    - [利用辅助栈](#利用辅助栈)
    - [基于差值存储最小值信息](#基于差值存储最小值信息)
  - [面试题03.05 栈排序](#面试题0305-栈排序)
  - [Offer 31. 栈的压入、弹出序列](#offer-31-栈的压入弹出序列)
  - [739. 每日温度 [Medium]](#739-每日温度-medium)
    - [利用单调栈解题](#利用单调栈解题)
  - [20. 有效的括号 [Easy]](#20-有效的括号-easy)
  - [150. 波兰表达式 [Easy]](#150-波兰表达式-easy)
  - [栈与DFS](#栈与dfs)
    - [与BFS的差异](#与bfs的差异)
    - [DFS 模版1 - 基于递归实现：](#dfs-模版1---基于递归实现)
    - [DFS 模版2 - 显式栈实现](#dfs-模版2---显式栈实现)
  - [200. 岛屿数量 [DFS实现]](#200-岛屿数量-dfs实现)
  - [133. 克隆图 [Medium]](#133-克隆图-medium)
  - [139. 单词拆分](#139-单词拆分)
  - [329. 矩阵中的最长递增路径](#329-矩阵中的最长递增路径)
  - [494. 目标和 target sum [Medium]](#494-目标和-target-sum-medium)
  - [二叉树的中序遍历](#二叉树的中序遍历)
  - [1047. 删除字符串中的所有相邻重复项](#1047-删除字符串中的所有相邻重复项)
- [进阶训练](#进阶训练)
  - [232. 用栈实现队列 [Easy]](#232-用栈实现队列-easy)
  - [225. 用队列实现栈 [Easy]](#225-用队列实现栈-easy)
  - [394. 字符串解码 [Medium]](#394-字符串解码-medium)
  - [726. 原子的数量](#726-原子的数量)
  - [733. 图像渲染 [Easy]](#733-图像渲染-easy)
  - [542. 01矩阵 [Matrix] **](#542-01矩阵-matrix-)
  - [547. 省份数量](#547-省份数量)
  - [841. 钥匙和房间](#841-钥匙和房间)
- [小结](#小结)
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


### 622. 设计循环队列 [Medium]*
- 利用数组进行队列数据管理，出队操作可以从原来的O(N)降至O(1)
- 参考代码中设计队列容量为 `k+1`
  - 主要是为了留出一个位置，用来判断队列是否满；
  - rear更严格的定义为：下一个元素插入的位置，但目前并未插入
  - 当`(rear+1)==k+1`时，数组实际存储了k个元素，当`(rear+1)%(k+1)==front`时说明数组已满，这种写法避免了最开始front和rear都为0时的特殊情况
- 解析参考： https://zhuanlan.zhihu.com/p/79163010
```c++
class MyCircularQueue {
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

```c++
class Solution {
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
### 695. 岛屿的最大面积
> 给定一个包含了一些 0 和 1 的非空二维数组 grid 。一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。**找到给定的二维数组中最大的岛屿面积**


- 与[LC200.岛屿数量]的最大区别在于多了一个统计岛屿面积，在bfs/dfs遍历时进行记录即可
- 遍历逻辑不变，都是每次遍历一个点，将其置为0
  - dfs 时间复杂度 O(mn) 空间复杂度 O(mn)
  - bfs 时间复杂度 O(mn) 空间复杂度 O(min(m,n))
- 关键点： **`dfs/bfs 的基本模板`**

```c++
class Solution {
public:
    int dfs(vector<vector<int>>& grid, int row, int col) {
        grid[row][col] = 0;
        int ans = 1;
        if (row - 1 >= 0 && grid[row - 1][col]) ans += dfs(grid, row - 1, col);
        if (row + 1 < grid.size() && grid[row + 1][col]) ans += dfs(grid, row + 1, col);
        if (col - 1 >= 0 && grid[row][col - 1]) ans += dfs(grid, row, col - 1);
        if (col + 1 < grid[0].size() && grid[row][col + 1]) ans += dfs(grid, row, col + 1);
        return ans;
    }
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int ans = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j]) {
                    ans = max(ans, dfs(grid, i, j));
                }
            }
        }
        return ans;
    }
};
```

### 1254. 统计封闭岛屿的数目
> 二维矩阵 grid ，每个位置要么是陆地（记号为 0 ）要么是水域（记号为 1 ）。
> 一座岛屿 完全 由水域包围，即陆地边缘上下左右所有相邻区域都是水域，那么我们将其称为 「封闭岛屿」
> 请返回封闭岛屿的数目

- 注意题目条件： 相邻区域均为水域即1，才构成。
  - 因此矩阵的边界行/列不能出现在当前搜索的0区域中
- 基于dfs进行搜索判断，
  - 对于边界添加额外处理，判断是否搜到了行/列的边界
```c++
class Solution {
public:
    vector<vector<int>> visited;
    int direct[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    void dfs(vector<vector<int>>& grid, int row, int col, bool& flag) {
        for (int i = 0; i < 4; i++) {
            int _row = row + direct[i][0];
            int _col = col + direct[i][1];
            if (_row >= 0 && _row < grid.size() && 
                _col >= 0 && _col < grid[0].size() &&
                visited[_row][_col] == 0 && grid[_row][_col] == 0) {
                    visited[_row][_col] = 1;
                    // 边界条件判断
                    if (_col == 0 || _row == 0 || _col == grid[0].size() - 1|| _row == grid.size() - 1) {
                        flag = false;
                    }
                    dfs(grid, _row, _col, flag);
            }
        }
        
    }
    int closedIsland(vector<vector<int>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();
        int ans = 0;
        visited.assign(grid.begin(), grid.end());
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols -1; j++) {
                if (grid[i][j] == 0 && visited[i][j] == 0) {
                    //cout << i << " " << j << endl;
                    visited[i][j] = 1;
                    bool flag = true;
                    dfs(grid, i, j, flag);
                    //cout << flag << endl;
                    if (flag) ans++;

                }
            }
        }
        return ans;
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

### 279. 完全平方数 [Medium]
#### BFS套模板
- 寻找最少的完全平方数数量构成target
- 题目描述反映出这可以通过BFS无脑求解
- 直接套用BFS模板，即可。
  - 在队列存储设置上，每步存储n-x的结果
  - 如果存储x+j从小到大的过程，遍历步数将与暴力搜索无异。。。。
  - 时间复杂度分析：取决于遍历深度 O($n^{h/2}$) h为遍历深度； 空间复杂度：取决于队列长度，即单层最大节点数量 $O(\sqrt{n}^h)$

```class Solution {
public:
    int numSquares(int n) {
        queue<int> q;
        q.push(n);
        int step = 0;
        vector<int> squares;
        for (int i = 1; i <= int(sqrt(n)); i++) {
            squares.push_back(i*i);
        }
        while (!q.empty()) {
            int s = q.size();
            for (int j = 0; j < s; j++){
                int cur = q.front();
                //cout << cur <<endl;
                q.pop();
                for (auto _s : squares) {
                    if (cur == _s)
                        return step + 1;
                }
                for (auto _s : squares) {
                    int tmp = cur - _s;
                    if (tmp > 0)
                        q.push(tmp);
                }
            }
            step++;
        }
        return step;

    }
};
```
- 优化：利用set代替queue来实现队列，减少重复遍历

```class Solution {
public:
    int numSquares(int n) {
        set<int> q;
        q.insert(n);
        int step = 0;
        vector<int> squares;
        for (int i = 1; i <= int(sqrt(n)); i++) {
            squares.push_back(i*i);
        }
        while (!q.empty()) {
            int s = q.size();
            set<int> queue_set;
            for (auto cur : q){
                for (auto _s : squares) {
                    if (cur == _s)
                        return step + 1;
                }
                for (auto _s : squares) {
                    int tmp = cur - _s;
                    if (tmp > 0)
                        queue_set.insert(tmp);
                }
            }
            step++;
            q = queue_set;
        }
        return step;

    }
};
```

#### dp解法
- dp[i] = min(dp[i-k]+1)
- 时间复杂度： $O(n*\sqrt{n})$ , 空间复杂度 O(n)
  
```class Solution {
public:
    int numSquares(int n) {
        vector<int> dp(n + 1);
        dp.assign(n + 1, INT_MAX);
        vector<int> squares;
        for (int i = 1; i <= sqrt(n); i++) {
            squares.push_back(i * i);
        }
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            for (auto k : squares) {
                if (i - k < 0)
                    break;
                if (dp[i - k] + 1 < dp[i])
                    dp[i] = dp[i - k] + 1;
            }
        }
        return dp[n];
    }
};
```

### 队列题目小结
- 常用于寻找最短路径及其延伸问题，牢记两种模板即可
- 往往题目不止BFS一种解法，可能还是DFS/DP的解法


## 栈 stack
### 基础知识
后入先出的结构，主要操作：
- push 入栈
- pop 出栈
- top 返回栈顶值
  
### 基本实现
利用STL：vector即可实现：
```class MyStack {
    private:
        vector<int> data;               // store elements
    public:
        /** Insert an element into the stack. */
        void push(int x) {
            data.push_back(x);
        }
        /** Checks whether the queue is empty or not. */
        bool isEmpty() {
            return data.empty();
        }
        /** Get the top item from the queue. */
        int top() {
            return data.back();
        }
        /** Delete an element from the queue. Return true if the operation is successful. */
        bool pop() {
            if (isEmpty()) {
                return false;
            }
            data.pop_back();
            return true;
        }
};
```
### 主要API
stack<int> s;
s.push(x);
s.pop();
s.top();
s.empty();
### 155. 最小栈 [Easy]
#### 利用辅助栈
- 用辅助栈来存储最小值信息

```
class MinStack {
private:
    vector<int> data;
    vector<int> min_data;
public:
    /** initialize your data structure here. */
    MinStack() {

    }
    
    void push(int x) {
        data.push_back(x);
        if (!min_data.empty() && x >= min_data.back()) {
                min_data.push_back(min_data.back());
        }
        else {
            min_data.push_back(x);
        }
        //cout << min_data.back() << endl;
    }
    
    void pop() {
        if (!data.empty()) {
            data.pop_back();
            min_data.pop_back();
            }

    }
    
    int top() {
        return data.back();
    }
    
    int getMin() {
        return min_data.back();

    }
};
```

#### 基于差值存储最小值信息
- 注意数值范围，**避免越界**
- 使用long 存储diff
  - 差值设计： _min保存以当前栈顶元素下的最小值
    - 初始化： _min = cur; diff = 0
    - 当插入元素大等于 _min时， 直接插入 `cur-_min` > 0
    - **当插入元素小于 _min时**， 要更新最小值，并插入diff
  - 出栈时
    - 当栈顶元素大/等于0， `diff + _min`即可得到答案
    - 否则，说明遇到分界点： 要恢复之前的最小值： `_min - diff` 
      - 因为之前的最小值大于目前栈顶最小值，存的`diff<0`, 直接`_min - diff`即可
  - 取栈顶时也是同样操作： `diff >= 0 ? _min + diff : _min` 
    - 因为对于分界位置： **当前值存在最小值上，而diff值保存的是当前值跟之前的最小值的差值**
```c++
class MinStack {
private:
    stack<long> st;
    long  _min;
    
public:
    /** initialize your data structure here. */
    MinStack() {

    }
    
    void push(int x) {
        if (!st.empty()) {
            long long diff = x - _min;
            _min = diff > 0 ? _min : x;
            st.push(diff);
        }
        else {
            _min = x;
            st.push(0);
        }
        //cout << min_data.back() << endl;
    }
    
    void pop() {
        if (!st.empty()) {
            long long diff = st.top();
            st.pop();
            if (diff < 0) {
                _min = int (_min - diff);
            }
        }
    }
    
    int top() {
        return st.top() < 0 ? _min : int(st.top() + _min);
    }
    
    int getMin() {
        return _min;
    }
};

```

### 面试题03.05 栈排序
> **对栈进行排序使最小元素位于栈顶**。最多只能使用一个其他的临时栈存放数据，但不得将元素复制到别的数据结构（如数组）中。该栈支持如下操作：push、pop、peek 和 isEmpty。当栈为空时，peek 返回 -1

- 设计题， 与最小栈题目相似，需要借助辅助栈实现
- 本题使用双栈来完成排序
  - less栈： **升序存储 （栈顶最大）**
  - greater栈： **降序存储（栈顶最小）**
  - **入栈时判断目标值小于less栈顶/还是大于greater栈顶，然后动态的移动两个栈**
  - 最后以greater为主栈进行pop/peek操作
  - 各类操作中都需要判断栈是否为空

```c++
class SortedStack {
public:
    stack<int> less; // 小于val
    stack<int> greater; // 大于val
    SortedStack() {

    }
    
    void push(int val) {
        // 大 -》 小 降序
        while (!less.empty() && val < less.top()) {
            greater.push(less.top()); // 将栈顶元素加入 greater
            less.pop();
        }
        // 小 -》 大 逆序
        while (!greater.empty() && val > greater.top()) {
            less.push(greater.top());
            greater.pop();
        }
        // 选择一个放入即可
        greater.push(val);
    }
    
    void pop() {
        // 将所有元素都加到greater上，构成自顶而下升序的栈
        while (!less.empty()) {
            greater.push(less.top());
            less.pop();
        }
        if (!greater.empty())
            greater.pop();
    }
    
    int peek() {
        while (!less.empty()) {
            greater.push(less.top());
            less.pop();
        }
        return greater.empty() ? -1 : greater.top();
    }
    
    bool isEmpty() {
        return less.empty() && greater.empty();
    }
};
```
### Offer 31. 栈的压入、弹出序列
> 第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等

- **栈先入后出性质的检验**
- 思路： 利用辅助栈进行出入栈过程模拟
  - 将入栈数组压入栈中，然后逐步判断栈顶元素是否等于当前出栈数组的目标元素
  - 若等于，则进行出栈操作
  - 最后**判断辅助栈是否为空**
- 时间复杂度 o(N) 空间复杂度 O(N)
```c++
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        stack<int> st;
        int index = 0;
        int n = pushed.size();
        for (int i = 0; i < n; i++) {
            st.push(pushed[i]); // 先入栈
            // 循环出栈操作
            while (!st.empty() && index < n && st.top() == popped[index]) {
                st.pop();
                index++;
            }   
        }
        return st.empty();
    }
};
```



### 739. 每日温度 [Medium]
#### 利用单调栈解题
- 单调栈 https://blog.csdn.net/lucky52529/article/details/89155694
- 从栈底到栈顶数据单调递增或递减。
- 逻辑模板：以单调递减栈为例
```c++
stack<int> st
for (visit list) {
    if (st空 ｜｜st.top >= 当前元素) 
        入栈
    else
        while (st不空 && st.top < 当前元素) 
        {
            出栈；
            update res;
        }
        当前数据入栈
}
```
- 直接套用单调栈模板得到结果
- 在实际答题中，需要能够发现单调栈的使用情况：
  - **要寻找任一个元素的右边或者左边第一个比自己大或者小的元素的位置**
```c++
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& T) {
        stack<int> st;
        vector<int> res(T.size(),0);
        for (int i = 0; i < T.size(); i++) {
            if (st.empty() || T[st.top()] >= T[i]) {
                st.push(i);
            } else {
                while (!st.empty() && T[st.top()] < T[i]) {
                    int prev = st.top();
                    res[prev] = i - prev;
                    st.pop();
                }
                st.push(i);
            }
        }
        return res;
    }
};
```

### 20. 有效的括号 [Easy]
> 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
**有效字符串需满足**：
左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。

- 判断字符串内括号是否有效，强调**同级括号的封闭性**
- 使用**栈**结构对字符串处理
- 主要思路是：**利用栈压入左字符，当遇到右字符时进行pop**
    - **考虑临界情况**：当字符串长度为奇时，返回false；
    - 在pop时考虑栈内是否有字符压入，即右字符先出现的特殊情况；
- 复杂度分析： 时间复杂度：O(N)   空间复杂度 O(N+C) *有哈希表的影响*
- 关键点： **`栈的基本操作`**

```c++
class Solution {
public:
    bool isValid(string s) {
        stack<char> st;
        unordered_map<char,char> chrs;
        chrs.insert({')','('});
        chrs.insert({']','['});
        chrs.insert({'}','{'});

        for(int i = 0; i < s.length(); i++) {
            if (chrs.count(s[i])) {
                if (st.empty() || st.top() != chrs[s[i]])
                    return false;
                st.pop();
            } 
            else {
                st.push(s[i]);
            }
        }
        return st.empty();
    }
};
```

### 150. 波兰表达式 [Easy]
- 栈的典型应用
```class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int> st;
        for (int i = 0; i < tokens.size(); i++) {
            if (tokens[i] == "-" || tokens[i] == "+" || tokens[i] == "*" || tokens[i] == "/") {
                int a = st.top();
                st.pop();
                int b = st.top();
                st.pop();
                if (tokens[i] == "-")
                    st.push(b - a);
                else if (tokens[i] == "+")
                    st.push(b + a);
                else if (tokens[i] == "*")
                    st.push(b * a);
                else if (tokens[i] == "/")
                    st.push(b / a);
            } 
            else {
                st.push(stoi(tokens[i]));
            }
        }
        return st.top();

    }
};
```
### 栈与DFS
使用栈来存储深度优先遍历过程中的遍历节点，并利用栈进行回溯。
#### 与BFS的差异
遍历顺序的差异导致，DFS达到目标点的第一次路径遍历不一定是最短路径

#### DFS 模版1 - 基于递归实现：
```
boolean DFS(Node cur, Node target, Set<Node> visited) {
    return true if cur is target;
    for (next : each neighbor of cur) {
        if (next is not in visited) {
            add next to visted;
            return true if DFS(next, target, visited) == true;
        }
    }
    return false;
}
```
#### DFS 模版2 - 显式栈实现
```
boolean DFS(int root, int target) {
    set<Node> visited;
    stack<Node> s;
    add root to s;
    while (!s.empty()) {
        Node cur = s.top();
        return true if cur is target;
        for (Node next : the neighbors of cur) {
            if (next is not in visited) {
                add next to s;
                add next to visited;
            }
        }
        s.pop();
    }
}
```







### 200. 岛屿数量 [DFS实现]
- 时间复杂度分析： O(MN); 空间复杂度：O(MN)（最差情况下，所有点都是1）
- BFS 时间复杂度: O(MN); 空间复杂度O(min(M,n))
```c++
class Solution {

public:
    void dfs(vector<vector<char>>& grid, int row, int col) {
        grid[row][col] = '0';
        int nr = grid.size();
        int nc = grid[0].size();
        if (row - 1 >= 0 && grid[row - 1][col] == '1')  dfs(grid, row - 1, col);
        if (row + 1 < nr && grid[row + 1][col] == '1')  dfs(grid, row + 1, col);
        if (col - 1 >= 0 && grid[row][col - 1] == '1')  dfs(grid, row, col - 1);
        if (col + 1 < nc && grid[row][col + 1] == '1')  dfs(grid, row, col + 1);   
    }
    int numIslands(vector<vector<char>>& grid) {
        int res = 0;
        if (grid.empty())
            return 0;
        int cols = grid[0].size();
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i, j);
                    res++;
                }
            }
        }
        return res;
    }
};
```

### 133. 克隆图 [Medium]

- 图遍历问题，BFS或者DFS都可以实现
- 主要在于构建新老节点的hash表来记录

```
/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
*/

class Solution {
public:
    unordered_map<Node*, Node*> visited;
    Node* dfs (Node* node) {
        if (node == nullptr)
            return node;
        if (visited.count(node)) {
            return visited[node];
        }
        Node* clone = new Node(node -> val);
        visited[node] = clone;
        for (auto& cur : node -> neighbors) {
            clone -> neighbors.emplace_back(dfs(cur));
        }
        return clone;
    }
    Node* cloneGraph(Node* node) {
        return dfs(node);
    }
};
```
BFS: 
```
class Solution {
public:
    unordered_map<Node*, Node*> visited;
    Node* cloneGraph(Node* node) {
        if (node == nullptr) 
            return node;
        stack<Node*> st;
        st.push(node);
        visited[node] = new Node(node -> val);
        while (!st.empty()) {
            int s = st.size();
            for (int i = 0; i < s; i++) {
                Node* cur = st.top();
                st.pop();
                for(auto& neigh : cur -> neighbors) {
                    if (!visited.count(neigh)) {
                        visited[neigh] = new Node(neigh -> val);
                        st.push(neigh);
                    }
                    visited[cur] -> neighbors.emplace_back(visited[neigh]);
                }
            }
        }
        return visited[node];
    }
};
```
### 139. 单词拆分
> 给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
说明：拆分时可以重复使用字典中的单词。你可以假设字典中没有重复的单词

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
```
- 字符串分割问题，暴力的思路就是枚举各种情况，那么再优化一点就是用回溯法来解题
  - 使用记忆化的思路来进行剪枝，**记录下以当位置开始子串是否能够被拆分**
  - 剩下的就是回溯法的模板
- 关键点： **`记忆化 + 回溯`**
- 时间复杂度 O(2^N) 空间复杂度 O(N)

```c++
class Solution {
public:
    bool backTrack(string s, unordered_set<string>& wordDict, int startIndex, vector<int>& memo) {
        if (startIndex == s.size()) return true;

        if (memo[startIndex] != -1) return memo[startIndex];
        for (int i = startIndex; i < s.size(); i++) {
            string word = s.substr(startIndex, i - startIndex + 1);
            if (wordDict.count(word) && backTrack(s, wordDict, i + 1, memo)) {
                memo[i] = 1;
                return true;
            }
        }
        memo[startIndex] = 0; // 以startIndex开始的子串不可分割
        return false;

    }
    bool wordBreak(string s, vector<string>& wordDict) {
        if (s.empty()) return true;
        vector<int> memo (s.size(), -1);
        unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
        return backTrack(s, wordSet, 0, memo);

    }
};
```
- 本题还有**动态规划**做法： 
  - 定义dp[i] 表示从[0,i]子串是否可以正确拆分
  - `dp[i] = (dp[i - j] & dp[j]);` 类似于0/1背包问题
  - 初始化： `dp<n+1,0>`
  - 遍历方向： 外层遍历不同位置[1,s.size()]； 内层遍历子段
- 时间复杂度 O(N^3) 内外循环+子段查找
```c++
class Solution {
public:
    
    bool wordBreak(string s, vector<string>& wordDict) {
        if (s.empty()) return true;
        vector<int> dp(s.size() + 1, 0);
        unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
        dp[0] = 1;
        // 从1开始编码， (j,i)取出的word刚好为目标子段
        for (int i = 1; i <= s.size(); i++) {
            for (int j = 0; j < i; j++) {
                string word = s.substr(j, i - j);
                // 判断当前词是否出现 且之前的值也出现过
                if (wordSet.count(word) && dp[j]) {
                    dp[i] = 1;
                }
            }
        }
        return dp[s.size()];

    }
};
```

### 329. 矩阵中的最长递增路径
> 一个 m x n 整数矩阵 matrix ，找出其中**最长递增路径**的长度。
对于每个单元格，你可以往上，下，左，右四个方向移动。 你不能在 对角线 方向上移动或移动到 边界外（即不允许环绕）。

- 基础思路： dfs/bfs， 从每个位置都搜一遍，但时间复杂度比较高； 为了降低复杂度，引入记忆化：
  - **记录每个位置上的最长递增路径值**
  - 当搜索中当前位置有值时，则不再向下搜索，直接取值即可
  - 因此遍历时返回当前位置记录的最长递增路径值即可
- 关键点： **`记忆化深度优先搜索`**

```c++
class Solution {
public:
    int dfs(vector<vector<int>>& matrix, int row, int col,  vector<vector<int>>& memo) {
        if (memo[row][col] != 0) 
            return memo[row][col];
        
        memo[row][col]++;
        if (row + 1 < matrix.size() && matrix[row + 1][col] > matrix[row][col])
            memo[row][col] = max(1 + dfs(matrix, row + 1, col, memo), memo[row][col]);
        if (row - 1 >= 0 && matrix[row - 1][col] > matrix[row][col])
            memo[row][col] = max(1 + dfs(matrix, row - 1, col, memo), memo[row][col]);
        if (col + 1 < matrix[0].size() && matrix[row][col + 1] > matrix[row][col])
            memo[row][col] = max(1 + dfs(matrix, row, col + 1, memo), memo[row][col]);
        if (col - 1 >= 0 && matrix[row][col - 1] > matrix[row][col])
            memo[row][col] = max(1 + dfs(matrix, row, col - 1, memo), memo[row][col]);
        return memo[row][col];
    }
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int ans = 1;
        vector<vector<int>> memo(matrix.size(), vector<int>(matrix[0].size()));
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix[0].size(); j++) {
                ans = max(ans, dfs(matrix, i, j, memo));
                //cout << dfs(matrix, i, j, memo) << endl;
            }
        }
        return ans;
    }
};
```

### 494. 目标和 target sum [Medium]
- 给定序列，计算仅利用+ -运算组合可以得到的目标值的组合数量
- DFS搜索所有可能组合，是最为暴力的解题方法
  - 时间复杂度： O(2^N),N为数组的长度，即组合的数量； 空间复杂度O(N)

```c++
class Solution {
public:
    int count = 0;
    void dfs(vector<int>& nums, int S, int cur, int i) {
        if (i == nums.size()) {
            if (cur == S)
            count++;
        }
        else 
        {
            dfs(nums, S, cur + nums[i], i+1);
            dfs(nums, S, cur - nums[i], i+1);
        }
    }
    int findTargetSumWays(vector<int>& nums, int S) {
        dfs(nums, S, 0, 0);
        return count;
    }
};
```
- 其他方法： 利用动态规划方法，大幅减少搜索空间
  - `dp[i][j]`定义i个数构成目标值j的组合数量
  - 状态转移方程：`dp[i][j] = dp[i-1][j-nums[i]] + dp[i-1][j+nums[i]]`
    - 初始化比较复杂： 由于j- nums[i]可能出现负值，因此需要将遍历空间调大： 增大到`[0,2*sum+1)`的范围；
    - 同时**初始化最开始的元素取值**：
      - `dp[0][nums[0]+sum]=1 dp[0][sum -nums[0]]=1`
      - 还要考虑当`nums[0]=0`的特殊情况，`dp[0][sum]=2`
    - 时间复杂度： O(N^2), 空间复杂度O(N*max(sum)) N为数组长度

```c++
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        vector<vector<int> > dp;
        int sum = 0;
        for (auto i: nums) {
            sum += i;
        }
        if (sum < S)
            return 0;
        for (int i = 0; i < nums.size(); i++) {
            vector<int> tmp;
            tmp.assign(2*sum + 1, 0);
            dp.push_back(tmp);
        }
        int t = 2*sum + 1;
        if (nums[0] == 0) {
            dp[0][sum] = 2;
        }
        else {
            dp[0][nums[0] + sum] = 1; // init 
            dp[0][sum - nums[0]] = 1; // init
        }
        //cout << t << endl;
        for (int i = 1; i < nums.size(); i++) {
            for (int j = 0; j < t; j++ ) {
                int l = j - nums[i] < 0 ? 0 : j - nums[i];
                int r = j + nums[i] >= t ? 0 : j + nums[i];
                //cout << l << r << endl;
                dp[i][j] = dp[i - 1][l] + dp[i - 1][r];
            }
        }
        return dp[nums.size() - 1][S + sum];
    }
};
```
- 在上面的基础上，可以对dp数组进行优化，仅利用两个一维数组进行存储。大幅降低空间复杂度
  - 由于上面的转移方程中仅涉及相邻状态的转换，因此可以利用两个数组进行相对更新存储，在动态规划中十分常见
```c++
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        vector<vector<int> > dp;
        int sum = 0;
        for (auto i: nums) {
            sum += i;
        }
        if (sum < S)
            return 0;
        
        vector<int> prev;
        prev.assign(2*sum + 1, 0);
        vector<int> next;
        next.assign(2*sum + 1, 0);
        int t = 2*sum + 1;
        if (nums[0] == 0) {
            prev[sum] = 2;
        }
        else {
            prev[nums[0] + sum] = 1; // init 
            prev[sum - nums[0]] = 1; // init
        }
        for (int i = 1; i < nums.size(); i++) {
            for (int j = 0; j < t; j++ ) {
                int l = j - nums[i] < 0 ? 0 : j - nums[i];
                int r = j + nums[i] >= t ? 0 : j + nums[i];
                next[j] = prev[l] + prev[r];
            }
            prev = next;
        }
        return prev[S + sum];
    }
};
```

### 二叉树的中序遍历

- 中序遍历：左节点->root->right
- 最基本的实现： 递归方式；
  
```
class Solution {
public:
    vector<int> vals;
    void inorder(TreeNode* root) {
        if (root == nullptr) 
            return ;
        inorder(root->left);
        vals.push_back(root -> val);
        inorder(root->right);
    }
    vector<int> inorderTraversal(TreeNode* root) {
        inorder(root);
        return vals;
 
    }
};
```
- 基于模板二的迭代实现：
按照上面的思路，利用stack进行迭代，每次遍历至最左侧的节点，然后进行结果记录，并开始右侧节点迭代。
```
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;
        vector<int> vals;
        while (!st.empty() || root) {
            while (root) {
                st.push(root);
                root = root -> left;
            }
            TreeNode* top = st.top();
            st.pop();
            vals.push_back(top -> val);
            root = top -> right;
        }
        return vals;
    }
};
```

### 1047. 删除字符串中的所有相邻重复项
> 给出由小写字母组成的字符串 S，重复项删除操作会选择两个相邻且相同的字母，并删除它们。
在 S 上反复执行重复项删除操作，直到无法继续删除。
在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

```
输入："abbaca"
输出："ca"
```
- 典型的栈结构利用
- 注意临界情况判断即可
```c++
class Solution {
public:
    string removeDuplicates(string S) {
        vector<char> ans;
        int index = 0;
        while (index < S.size()) {
            if (ans.empty()) {
                ans.push_back(S[index]);
            }
            else {
                bool flag = false;
                while (!ans.empty() && S[index] == ans.back()) {
                    ans.pop_back();
                    flag = true;
                }
                if (!flag) ans.push_back(S[index]); 
            }
            index++;
        }
        string res = "";
        for (auto s : ans) {
            res.push_back(s);
        }
        return res;
    }
};
```


## 进阶训练

### 232. 用栈实现队列 [Easy]
- 构建两个栈来存储数据，以实现从FILO到FIFO的转换
- 一个栈做输入栈，push入数据；另一个栈做输出栈，将输入栈的数据逐个从顶弹出再压入输出栈，从而完成了顺序的变化
- 时间复杂度分析：均摊到O(1) 空间复杂度O(N)
```
class MyQueue {
public:
    stack<int> in_st;
    stack<int> out_st;
    /** Initialize your data structure here. */
    MyQueue() {

    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        in_st.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        int res = peek();
        if (!out_st.empty()) {
            res = peek();
           out_st.pop();
           return res;
        }
        return res;
    }
    
    /** Get the front element. */
    int peek() {
        if (out_st.empty()) {
            while (!in_st.empty()) {
                out_st.push(in_st.top());
                in_st.pop();
            }
        }  
        if (!out_st.empty()){
            int res = out_st.top();
            return res;
        }
        return -1;
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        if (in_st.empty() &&  out_st.empty()) {
            return true;
        }
        return false;
    }
};
```

### 225. 用队列实现栈 [Easy]
- 跟上题相似，通过双队列交换实现栈
- 用一个队列存储当前最新的输入，并加另一个临时队列存储的输入，加入队尾；
- 通过这种迭代实现， 临时队列中的存储是LIFO的结构，最近push进入的永远在队首
- 时间复杂度： 入栈O(n) 空间复杂度: O(n)

```c++
class MyStack {
    queue<int> q1;
    queue<int> q2;
public:
    /** Initialize your data structure here. */
    MyStack() {

    }
    
    /** Push element x onto stack. */
    void push(int x) {
        q1.push(x);
        while (!q2.empty()) {
            q1.push(q2.front());
            q2.pop();
        }
        swap (q1, q2);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int res = q2.front();
        q2.pop();
        return res;
    }
    
    /** Get the top element. */
    int top() {
        int res = q2.front();
        return res;
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
        if (q2.empty())
            return true;
        return false;
    }
};
```
- 进一步优化,仅利用**一个队列**，每次push时都进行队列内元素后移，相当于进行插入操作
```
class MyStack {
    queue<int> q1;

public:
    /** Initialize your data structure here. */
    MyStack() {

    }
    
    /** Push element x onto stack. */
    void push(int x) {
        int n = q1.size();
        q1.push(x);
        for (int i = 0; i < n; i++) {
            q1.push(q1.front());
            q1.pop();
        }
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int res = q1.front();
        q1.pop();
        return res;
    }
    
    /** Get the top element. */
    int top() {
        int res = q1.front();
        return res;
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
        if (q1.empty())
            return true;
        return false;
    }
};
```
  
### 394. 字符串解码 [Medium]
> 输入：s = "3[a]2[bc]"
输出："aaabcbc"


- 思路比较简单，但在实现上细节比较多，细节是魔鬼
- 直接通过栈存储`]`之前的字母(string(1, char]))或`[`或者数字
  - 数字通过string遍历，获取完整数字范围 
- 遇到`]`时进行**出栈操作**
  - 先取出栈中`[`之内的字母，然后取出栈顶的数字  stoi()
  - 进行字符串拼接，将拼接后的字符串再压入栈中
- 当遍历完成后，将栈中的字符串拼接得到结果
- 时间复杂度分析： O(N)，空间复杂度O(N)
  

```c++
#include <algorithm>
class Solution {
public:
    string getDigit(string& s, size_t& ptr) {
        string res = "";
        while (ptr < s.size() && isdigit(s[ptr])) {
            res.push_back(s[ptr++]);
        }
        return res;
    }

    string getString(vector<string>& s) {
        string res = "";
        for(auto& _s: s) {
            res += _s;
        }
        return res;

    }
    string decodeString(string s) {
        size_t ptr = 0;
        vector<string> st;
        while(ptr < s.size()) {
            if (isdigit(s[ptr])) {
                st.push_back(getDigit(s,ptr));
            }
            else if (isalpha(s[ptr]) || s[ptr]=='[') {
                st.push_back(string(1, s[ptr++])); //  转换char to string
            }
            else {
                ptr++;
                // 出栈操作
                vector<string> tmp; //记录出栈字符串信息
                while (st.back()!="[") {
                    tmp.push_back(st.back());
                    st.pop_back();
                }
                st.pop_back();// [ 出栈
                int repTime = stoi(st.back());
                st.pop_back();
                reverse(tmp.begin(), tmp.end());
                string cur = getString(tmp);
                string _c = cur;
                for (int i = 0; i < repTime - 1; i++) {
                    cur += _c;
                }
                st.push_back(cur);
            }
        }
        string res = "";
        for (auto s: st) {
            res += s;
        }
        return res;
    }
};
```

- 基于递归思想实现
  - 递归思考起来跟前面的栈方式基本差不多
  - 但是很难写。。。。。 栈形式相较而言还是能写出来的
    - 递归终止条件
    - 递归返回情况
    - 字符串内部处理分类： 字母/数字

```c++
#include <algorithm>
class Solution {
    string src;
    size_t ptr;
public:
    string getDigit() {
        string res = "";
        while (ptr < src.size() && isdigit(src[ptr])) {
            res.push_back(src[ptr++]);
        }
        return res;
    }

    string getString() {
        // 终止条件
        if (ptr == src.size() || src[ptr] == ']') {
            return "";
        }
        string ret = "";
        if (isdigit(src[ptr])) {
            int repTime = stoi(getDigit()); //获取重复次数
            ptr++; // 跳过 [
            string cur = getString(); // 构建字符串
            ptr++; // 跳过 ]
            // 重复字符串
            while (repTime--) 
                ret += cur;
        }
        else if (isalpha(src[ptr])) {
            ret += string(1, src[ptr++]);// 拼接字符串
        }
        return ret + getString(); //对递归结果进行拼接，以得到完整的结果
    }
    string decodeString(string s) {
        src = s;
        ptr = 0;
        return getString();
    }
};
```
###  726. 原子的数量

```
输入：formula = "Mg(OH)2"
输出："H2MgO2"
解释： 
原子的数量是 {'H': 2, 'Mg': 1, 'O': 2}。
```

- 与[LC394字符串解码]相似，通过栈进行嵌套结构的处理，每层使用哈希表进行存储
  - 遇到`(`时插入空哈希表
  - 遇到`)`进行出栈操作，取当前栈顶哈希表，把对应元素数量加到结果上即可
  - 其他情况进行字母和数字的扫描判断即可
    - 需要注意数字的情况:对于`(H)`的情况，要注意返回1
- 关键点：**`哈希表+栈结构`**
```c++
class Solution {
public:
    string getAtom(string& s, int& idx) {
        string res;
        res.push_back(s[idx]);
        idx++;
        while (idx < s.size() && s[idx] <= 'z' && s[idx] >= 'a') {
            res.push_back(s[idx]);
            idx++;
        }
        return res;
    }
    int getNum(string& s, int& idx) {
        string res;
        while ( idx < s.size() && isdigit(s[idx])) {
            res.push_back(s[idx]);
            idx++;
        }
        if (res.empty()) {
            return 1;
        }
        return stoi(res);
    }
    string countOfAtoms(string formula) {
        stack<unordered_map<string, int>> st;
        int idx = 0;
        string tmp = "";
        st.push({}); // 使用{}来插入空哈希表
        while (idx < formula.size()) {
            if (isalpha(formula[idx])) {
                 tmp = getAtom(formula, idx);
                 if (isdigit(formula[idx])) {
                     int num = getNum(formula, idx);
                     st.top()[tmp] += num;
                 }
                 else {
                     st.top()[tmp] += 1; 
                 }
            }
            else if (formula[idx] == '(') {
                st.push({});
                idx++;
            }
            else if (formula[idx] == ')') {
                idx++;
                int num = getNum(formula, idx);
                unordered_map<string, int> t = st.top();
                st.pop();
                for (auto& iter : t) {
                    st.top()[iter.first] += iter.second * num;
                }
            }
        }
        vector<pair<string, int>> res(st.top().begin(), st.top().end());
        sort(res.begin(), res.end());
        string ans = "";
        for (auto& p : res) {
            ans += p.first;
            if (p.second > 1)
                ans += to_string(p.second);
        }
        return ans;
    }
};
```


### 733. 图像渲染 [Easy]
- DFS 需要重点考虑初始节点与目标颜色相同的情况
```
class Solution {
public:
    void dfs(vector<vector<int>>& image, int sr, int sc, int oldColor, int newColor) {
        int rows = image.size();
        int cols = image[0].size();
        image[sr][sc] = newColor;
        if (sr - 1 >= 0 && image[sr - 1][sc] == oldColor) dfs(image, sr - 1, sc, oldColor, newColor);
        if (sr + 1 < rows && image[sr + 1][sc] == oldColor) dfs(image, sr + 1, sc, oldColor, newColor);
        if (sc - 1 >= 0 && image[sr][sc - 1] == oldColor) dfs(image, sr, sc - 1, oldColor, newColor);
        if (sc + 1 < cols && image[sr][sc + 1] == oldColor) dfs(image, sr, sc + 1, oldColor, newColor);
        return;
    }
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
        if (image[sr][sc] != newColor) {
            dfs(image, sr, sc, image[sr][sc], newColor);
        }
        return image;
    }
};
```

- BFS: **不同的image渲染时间对程序效率影响很大，在对当前队首四周节点进行遍历时就进行image修改，可以很大程度减少遍历次数**。 如果每次仅在队列取首时才进行修改，会有很多重复遍历

```
class Solution {
public:
    void bfs(vector<vector<int>>& image, int sr, int sc, int oldColor, int newColor) {
        queue<pair<int, int> > q;
        q.push({sr,sc});
        image[sr][sc] = newColor;
        int rows = image.size();
        int cols = image[0].size();
        while (!q.empty()) {
            int q_size = q.size();
            for (int i = 0; i < q_size; i++) {
                pair<int, int> cur = q.front();
                q.pop();
                int row = cur.first;
                int col = cur.second;
                if (row - 1 >= 0 && image[row - 1][col] == oldColor) {
                    q.push({row - 1, col});
                    image[row - 1][col] = newColor;
                }
                if (row + 1 < rows && image[row + 1][col] == oldColor) {
                    q.push({row + 1, col});
                    image[row + 1][col] = newColor;
                }
                if (col - 1 >= 0 && image[row][col - 1] == oldColor) {
                    q.push({row, col - 1});
                    image[row][col - 1] = newColor;
                }
                if (col + 1 < cols && image[row][col + 1] == oldColor) {
                    q.push({row, col + 1});
                    image[row][col + 1] = newColor;
                }
            }
        }
    }
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
        if (image[sr][sc] != newColor) {
            bfs(image, sr, sc, image[sr][sc], newColor);
        }
        return image;
    }
};
```
###  542. 01矩阵 [Matrix] **

- 巧妙地对问题进行分析，在计算每个位置上的答案时不是重复搜索，而是从0出发反向搜索，一次全图的bfs就得到了所有位置上到0的距离
  - 时间复杂度：O(rows*cols) 空间复杂度 O(rows * cols)
  - 初始设置：设置visited矩阵记录已访问的信息；将0元素位置首先加入队列中
```c++
class Solution {

public:
    int dirs_x[4] = {-1, 1, 0, 0};
    int dirs_y[4] = {0, 0, -1, 1};

    vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        queue<pair<int, int> > q;
        vector<vector<int> > res(rows, vector<int>(cols));
        vector<vector<int> > visited(rows, vector<int>(cols)); //记录访问信息
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == 0) {
                    visited[i][j] = 1;
                    q.push({i,j});
                }
            }
        }

        while (!q.empty()) {
            int q_size = q.size();
            for (int i = 0; i < q_size; i++) {
                pair<int, int> cur = q.front();
                q.pop();
                int row = cur.first;
                int col = cur.second;
                for (int j = 0; j < 4; j++) {
                    if (row + dirs_x[j] >= 0 && row + dirs_x[j] < rows && 
                         col + dirs_y[j] >= 0 && col + dirs_y[j] < cols && 
                         !visited[row + dirs_x[j]][col + dirs_y[j]]) {
                          q.push({row + dirs_x[j], col + dirs_y[j]});
                          visited[row + dirs_x[j]][col + dirs_y[j]] = 1;
                          res[row + dirs_x[j]][col + dirs_y[j]] = res[row][col] + 1;
                         }
                }

            }
        }
        return res;
    }
};
```
- **可以进一步将上面的visited矩阵省略，仅判断当前位置的结果以及是否为1即可避免重复遍历哦**
  
```class Solution {

public:
    int dirs_x[4] = {-1, 1, 0, 0};
    int dirs_y[4] = {0, 0, -1, 1};

    vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        queue<pair<int, int> > q;
        vector<vector<int> > res(rows, vector<int>(cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == 0) {
                    q.push({i,j});
                }
            }
        }

        while (!q.empty()) {
            int q_size = q.size();
            for (int i = 0; i < q_size; i++) {
                pair<int, int> cur = q.front();
                q.pop();
                int row = cur.first;
                int col = cur.second;
                for (int j = 0; j < 4; j++) {
                    if (row + dirs_x[j] >= 0 && row + dirs_x[j] < rows && 
                         col + dirs_y[j] >= 0 && col + dirs_y[j] < cols && 
                         !res[row + dirs_x[j]][col + dirs_y[j]] &&
                         matrix[row + dirs_x[j]][col + dirs_y[j]]) {
                          q.push({row + dirs_x[j], col + dirs_y[j]});
                          
                          res[row + dirs_x[j]][col + dirs_y[j]] = res[row][col] + 1;
                         }
                }

            }
        }
        return res;
    }
};
```

### 547. 省份数量
> 给出城市之间的连通矩阵，省份定义是一组直接或间接相连的城市，组内不含其他没有相连的城市。计算省份数量


- 经典的dfs/bfs搜索任务
  - 通过dfs/bfs搜索最大连通部分，每次搜索都记录已遍历过的城市
  - 本质上与LC200.岛屿数量基本一致，本题更加简单，只有N个城市，N*N的矩阵
- 关键点： **`dfs/bfs`** **`visited数组`**
- 时间复杂度： O(N^2) 空间复杂度O(N)
```c++
class Solution {
public:
    
    void dfs(vector<vector<int>>& isConnected, int k, vector<int> & visited) {
        visited[k] = 1;
        for (int i = 0; i < isConnected[k].size(); i++) {
            if (isConnected[k][i] && i != k && visited[i] == 0) {
                visited[i] = 1;
                dfs(isConnected, i, visited);
            }
        }
    }
    int findCircleNum(vector<vector<int>>& isConnected) {
        int ans = 0;
        vector<int> visited(isConnected.size());
        for (int i = 0; i < isConnected.size(); i++) {
            if (visited[i] == 0) {
                ans++;
                dfs(isConnected, i, visited);
            }
        }
        return ans;
    }
};
```

### 841. 钥匙和房间
- DFS遍历，记录已访问过的节点，避免重复访问，同时可以记录遍历节点的数量；
```
class Solution {
public:
    vector<int> visited;
    void dfs(vector<vector<int>>& rooms, int x) {
        visited[x] = 1;
        for (int i = 0; i < rooms[x].size(); i++) {
            int next = rooms[x][i];
            if (!visited[next])
                dfs(rooms, next);
        }
    }
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        int N = rooms.size();
        visited.resize(N);
        dfs(rooms, 0);
        int res = 0;
        for (auto s : visited) 
            res +=s;
        return res == N; 
    }
};

```


## 小结
- 熟记DFS和BFS的基本模版，并考虑是否需要加visited
- 重复访问节点可以通过哈希表set/map进行实现，也可以通过array数组的 形式进行组织
- 对应清楚queue和stack在两者间的作用
- 关于DFS，递归模式要比较仔细


```
 vector<int> l_max;
        vector<int> r_max;
        l_max.push_back(0);
        int cur_max = heights[0];
        int cur_max_id = 0;
        for (int i = 1; i < heights.size(); i++) {
            if (heights[i] >= cur_max) {
                cur_max = heights[i];
                l_max.push_back(i); //当前最高
                cur_max_id = i;
            } 
            else {
                l_max.push_back(cur_max_id);
            }
        }
        int len = heights.size();
        r_max.push_back(len - 1);
        cur_max = heights[len - 1];
        cur_max_id = len - 1;
        for (int i = len - 2; i >= 0; i--) {
            if (heights[i] >= cur_max) {
                cur_max = heights[i];
                r_max.push_back(i); //当前最高
                cur_max_id = i;
            } 
            else {
                r_max.push_back(cur_max_id);
            }
        }
        reverse(r_max.begin(),r_max.end());
       // for (auto i : l_max) 
        //    cout << i << " ";
        //for (auto i : r_max) 
        //    cout << i << " ";
        
        // 计算res
        vector<int> ans;
        ans.push_back(r_max[0] + 1);
        for (int i = 1; i < len - 1; i++) {
            if (r_max[i] == l_max[i]) {
                ans.push_back(r_max[i + 1]-l_max[i-1]);
            }
            else {
               ans.push_back(r_max[i + 1]-l_max[i-1]+1); 
            }
           
             
            
        }
        ans.push_back(len - l_max[len - 1]);
        return ans;
```