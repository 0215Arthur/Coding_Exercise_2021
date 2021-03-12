
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
  - [739. 每日温度 [Medium]](#739-每日温度-medium)
    - [利用单调栈解题](#利用单调栈解题)
  - [20. 有效的括号 [Easy]](#20-有效的括号-easy)
  - [150. 波兰表达式 [Easy]](#150-波兰表达式-easy)
  - [栈与DFS](#栈与dfs)
    - [与BFS的差异](#与bfs的差异)
    - [DFS 模版1](#dfs-模版1)
  - [200. 岛屿数量 [DFS实现]](#200-岛屿数量-dfs实现)
  - [133. 克隆图 [Medium]](#133-克隆图-medium)
  - [494. 目标和 target sum [Medium]](#494-目标和-target-sum-medium)
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
```
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

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
```

### 739. 每日温度 [Medium]
#### 利用单调栈解题
- 单调栈 https://blog.csdn.net/lucky52529/article/details/89155694
- 从栈底到栈顶数据单调递增或递减。
- 逻辑模板：以单调递减栈为例
```
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
```
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
- 对stack的基本使用

```
class Solution {
public:
    bool isValid(string s) {
        stack<char> st;
        unordered_map<char,char> chrs;
        chrs.insert({')','('});
        chrs.insert({']','['});
        chrs.insert({'}','{'});
        // {{')', '('},
        //     {']', '['},
        //     {'}', '{'}};

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

#### DFS 模版1
基于递归实现：
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
### 200. 岛屿数量 [DFS实现]
- 时间复杂度分析： O(MN); 空间复杂度：O(MN)（最差情况下，所有点都是1）
- BFS 时间复杂度: O(MN); 空间复杂度O(min(M,n))
```
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

### 494. 目标和 target sum [Medium]
- 给定序列，计算仅利用+ -运算组合可以得到的目标值的组合数量
- DFS搜索所有可能组合，是最为暴力的解题方法
  - 时间复杂度： O(2^N),N为数组的长度，即组合的数量； 空间复杂度O(N)

```
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

```
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
```
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