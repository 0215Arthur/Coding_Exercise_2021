
![](./summary.png)
动画图解：https://github.com/chefyuan/algorithm-base
[TOC]

### 比较排序 v.s. 非比较排序

常见的快速排序、归并排序、堆排序、冒泡排序等属于比较排序。在排序的最终结果里，元素之间的次序依赖于它们之间的比较。每个数都必须和其他数进行比较，才能确定自己的位置。

在冒泡排序之类的排序中，问题规模为n，又因为需要比较n次，所以平均时间复杂度为O(n²)。在归并排序、快速排序之类的排序中，问题规模通过分治法消减为logN次，所以时间复杂度平均O(nlogn)。
**比较排序的优势是，适用于各种规模的数据，也不在乎数据的分布，都能进行排序。可以说，比较排序适用于一切需要排序的情况。**

计数排序、基数排序、桶排序则属于非比较排序。非比较排序是通过确定每个元素之前，应该有多少个元素来排序。针对数组arr，计算arr[i]之前有多少个元素，则唯一确定了arr[i]在排序后数组中的位置。
**非比较排序只要确定每个元素之前的已有的元素个数即可，所有一次遍历即可解决。算法时间复杂度O(n)。非比较排序时间复杂度底，但由于非比较排序需要占用空间来确定唯一位置。所以对数据规模和数据分布有一定的要求。**


## 交换排序

### 0. 冒泡排序
- 特点：遍历进行对比交换，时间复杂度 O(N2), 稳定排序
- 基本盘

```
void bubbleSort(vector<int>& arr) {
    int L = arr.size();
    for (int i = 0; i < L; i++) {
        bool flag = false;
        for (int j = 0; j < L - 1 -i; j++) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
                flag  = true;
            }
        }
        cout << i << "th :";
        for (int k = 0; k < L; k++) {
            cout << arr[k] << " ";
        }
        cout << endl;
        if (!flag)
            break; 
    }
}
```

### 1.快速排序
- 特点：左右交换，递归进行, **非稳定排序**
- 空间复杂度 O(LogN)
- 时间复杂度 O(NLogN)  最差情况下 O(N^2)
```
int Parition( vector<int>& arr, int left, int right) {
    int pivot = arr[left];
    int start = left;
    while(left < right) {
        while(left < right && arr[right] >= pivot) {
            right--;
        }
        while (left < right && arr[left] <= pivot) {
            left++;
        }

        if (left < right) {
            swap(arr[left],arr[right]);
        }
    }
    swap(arr[start], arr[left]);
    return left;
}

void quickSort(vector<int>& arr, int left, int right) {
    if (left >= right)
        return;
    int p = Parition(arr,left,right);
    cout << p  << endl;
    quickSort(arr,left,p-1);
    quickSort(arr,p+1, right);
}
```

### 2. 选择排序

原理： 每次遍历选择最小/大的一个，n次遍历得到每个位置上的值，时间复杂度稳定在`O(N^2)`
- 时间复杂度： O(N^2)
- 空间复杂度:  O(1)
- 稳定性：  不稳定 

>序列5 8 5 2 9，第一遍选择第1个元素5会和2交换，那么原序列中2个5的相对前后顺序就被破坏了
```
void selectSort(vector<int>& arr) {
    int Len = arr.size();
    for (int i = 0; i < Len; i++) {
        for (int j = i + 1; j < Len; j++) {
            if (arr[j] < arr[i]) {
                swap(arr[j], arr[i]);
            }
        }
    }
}


```


### 3. 插入排序 
原理： 构建有序数组，每次往前面构成的有序数组中插入
- 时间复杂度： O(N^2)  最好的情况： O(N) 最差的情况： O(N^2)
- 空间复杂度:  O(1)
- 稳定性： 稳定 (不存在位置交换)

```
void insertSort(vector<int>& arr) {
    int Len = arr.size();
    for (int i = 0; i < Len - 1; i++) {
        int prev = i;
        int cur = arr[i+1];
        while (cur < arr[prev] && prev >= 0) {
            arr[prev + 1] = arr[prev];
            prev--;
        }
        arr[prev + 1] = cur;
    }
}
```
### 4. 希尔排序 [nlogn]
- 简单插入排序的升级
- 将整个数组按照gap分成不同的子集，每个子集进行插入排序，并逐步缩小gap值
- 时间复杂度比之前的O(N2)有缩小
  - 时间复杂度： O(NlogN)  最好的情况： O(N) 最差的情况： O(NlogN)
  - 空间复杂度:  O(1)
  - 稳定性： 稳定 (不存在位置交换)
```
void shellSort(vector<int>& arr) {
    int Len = arr.size();
    int gap = Len/2;
    while (gap>0) {
        for (int i = gap; i < Len; i++) {
            int cur = arr[i];
            int prev = i - gap;
            while ( cur < arr[prev] && prev>=0) {
                arr[prev + gap] = arr[prev];
                prev-=gap;
            }
            arr[prev+gap] =  cur;
        }
        gap/=2;
    }
}
```
 
### 5. 归并排序

### 6. 堆排序 
- HeapSort
- 基于完全二叉树的结构：
  - 大顶堆 [用于从小到大排序] 根节点的值大于两个子节点的值；
  - 小顶堆 [用于从大到小排序] 根节点的值小于两个子节点的值
- 排序过程：
  - 1. 构建大顶堆
  - 2. 取出arr[0]即顶元素跟尾部元素交换，然后对剩下的部分继续排序
  - 重复步骤2，得到有序的数组
- 时间复杂度 `O(nlogn)` 最好情况 `O(nlogn)` 最坏情况 `O(nlogn)`
- 空间复杂度 `O(1)`
- 非稳定排序
```
void adjustHead(vector<int> & arr, int Len, int index) {
    int maxIdx = index;
    int left = 2*index + 1;
    int right = 2*index + 2;
    // 进行节点管理
    if (left < Len && arr[left] > arr[maxIdx]) maxIdx = left;
    if (right < Len && arr[right] > arr[maxIdx]) maxIdx = right;
    if (maxIdx != index) { // 调整子树
        swap(arr[maxIdx], arr[index]);
        adjustHead(arr, Len, maxIdx); 
    } 
}

void heapSort(vector<int>& arr) {
    int Len = arr.size();
    // 构建大顶堆， 取所有非叶子节点进行遍历，自底而上
    for(int i = Len/2 - 1 ; i >= 0; i--) {
        adjustHead(arr, Len, i);
    }
    for (int i = 0; i < Len; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    cout << "sort" <<endl;
    for (int i = Len - 1; i > 0; i--) {
        swap(arr[0],arr[i]);  // 把最大值调换到最后
        adjustHead(arr, i, 0); // 从头开始调整堆
        for (int i = 0; i < Len; i++) {
            cout << arr[i] << " ";
        }
        cout << endl;
    }
}
```
- **基于优先队列**实现的堆排序
- STL 的priority_queue更方便，优先队列的底层就是一个堆结构，在优先队列中，队首元素一定是当前队列中优先级最高的那一个。
- 通过 top() 函数来访问队首元素（也可称为堆顶元素），也就是优先级最高的元素。

```
priority_queue< int, vector<int>, greater<int> > q;  // 小顶堆
priority_queue< int, vector<int>, less<int> > q;     // 大顶堆
```

```

```


## 排序实战
### 386. 整数的字典序 [Medium] [ByteDance]
- **字典序可以视为树结构**
- 首个数字不能是0，但其他位置可以是0，
  - 通过DFS对树结构进行遍历，保存结果
```
class Solution {
public:
    vector<int> ans;
    void dfs(int tmp, int target) {
        if (tmp > target)
            return;
        ans.push_back(tmp);
        for (int i = 0; i<= 9; i++) {
            dfs(tmp*10+i, target);
        }
    }
    vector<int> lexicalOrder(int n) {
        for(int i = 1; i <= 9; i++) {
            dfs(i, n);
        }
        return ans;
    }
};
```
- 迭代方式实现
- 注意点： 入栈顺序从大到小，而且要注意外层遍历的范围:`min(0,n)`

```
class Solution {
public:
    vector<int> ans;
    vector<int> lexicalOrder(int n) {
        stack<int> st;
        for(int i = min(9,n); i >= 1; i--) {
            st.push(i);
        }
        while(!st.empty()) {
            int cur = st.top();
            st.pop();
            ans.push_back(cur);
            for (int i = 9; i >=0; i--) {
                if (cur*10 + i <= n) 
                    st.push(cur*10 + i);
            }
            
        }
        return ans;
    }
};
```
### 75. 颜色分类 [Medium]
- 三种颜色 0 1 2 ， 要实现对这些颜色的原地排序 从0到1
- 双指针法： left指针控制0， right指针控制2，进行交换，只需要遍历一次即可完成排序
    - 时间复杂度： `O(N)`

```
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int left = 0;
        int right = nums.size() - 1;
        for (int i = 0; i <= right; i++){
            if (nums[i] == 0) {
                swap(nums[i], nums[left]);
                left++;
            }
            if (nums[i] == 2) {
                swap(nums[i], nums[right]);
                right--;
                i--;// 解题关键
            }
            // for (auto s : nums) {
            //     cout << s << " ";
            // }
            // cout << endl;
        }
    }
};
```

### 347. 前K个高频元素 [Medium]*
- 计算出现次数最高的K个元素，返回对应的数组
##### 基于快速排序的思想
  - 核心是快速定位到K大元素的区间
  - 因为每次寻找是大于/等于 pivot的元素，并移动到左区间
  - 对于小于pivot的元素，本题中不用考虑，因此可以大幅降低时间开销
  - 然后递归时同样仅考虑`K`大所在子区间，不用每个分区间都考虑
  - 时间复杂度： O（N）

```
class Solution {
public:
    void quickSort(vector<pair<int, int>>& nums, int k, int left, int right, vector<int>& res) {
        int start = left;
        int pivot = nums[start].second;
        for (int i = left + 1; i <= right; i++) {
            if(nums[i].second > pivot) {
                swap(nums[i], nums[left + 1]);
                left++;
            }
        }
        swap(nums[left],nums[start]);

        // 判断递归方向，重要
        if (k <= left - start) {
            quickSort(nums, k, start, left - 1, res);
        }
        else {
            for (int i = start; i <= left; i++){
                res.push_back(nums[i].first);
            }
            if (k > left - start + 1) {
                quickSort(nums, k - (left - start + 1), left + 1, right, res);
            }
        }
    }
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> occurs;
        for (int i = 0; i < nums.size(); i++) {
            occurs[nums[i]]++;
        }
        vector<pair<int, int>> freqs;
        for (unordered_map<int,int> ::iterator iter = occurs.begin(); iter != occurs.end(); iter++) {
            freqs.push_back({iter->first, iter->second});
        }
        vector<int> ans;
        quickSort(freqs, k, 0, freqs.size() - 1, ans );
        return ans;

    }
};
```

##### 基于堆的思想
- 要取最大的K个值，需要构建小顶堆，存储最大的K个值
- 可以通过优先队列来实现堆结构，当队列长度等于K时
  - 需要进行判断，判断当前堆顶元素是否小于目标值，若小于需要弹出堆
  - 通过重载运算符来实现小顶堆

```
class Solution {
public:

    // static bool cmp(pair<int, int>& m, pair<int, int>& n) {
    //         return m.second > n.second;
    //     }

    struct cmp {
        bool operator()(const pair<int,int> a, const pair<int, int> b) {
            
            return a.second > b.second;
        }
    };
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> occurs;
        for (int i = 0; i < nums.size(); i++) {
            occurs[nums[i]]++;
        }
       priority_queue<pair<int,int>, vector<pair<int,int>>, cmp> q;
        //priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(&cmp)> q(cmp);

       for (auto & [num, count] : occurs) {
           if (q.size() == k) {
               if (q.top().second < count) {
                   q.pop();
                   q.emplace(num,count);
               }
           }
           else {
               q.emplace(num,count);
           }
       } 

       vector<int> ans;
        while (!q.empty()) {
            ans.push_back(q.top().first);
            q.pop();
        }
        return ans;
    }
};
```

### 215. 数组中的第K个最大元素 [Medium]*

- 基于堆的排序方法
  - 使用优先队列代替实现
  - 构建最小堆
  - 最后返回堆顶元素即可
  - **时间复杂度： O(NlogN) 空间复杂度 `O(logN)` 递归处理堆的过程**
```
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        // 定义最小堆
        priority_queue<int, vector<int>, greater<int> > q;
        for (int i = 0; i < nums.size(); i++) {
            if (q.size() == k) {
                if (q.top() < nums[i]) {
                    q.pop();
                    q.push(nums[i]);
                }
            } else{
                q.push(nums[i]);
            }
        }
        
        return q.top();
       
    }
};
```
- 基于快速排序的思路
  - 与上一题思路基本一致，只要在指定区域进行搜索即可
  - 时间复杂度: O(N) 空间复杂度:`O(logN)` 递归栈的深度

```
class Solution {
public:
    void quickSort(vector<int>& nums, int left, int right, int k, vector<int>& res) {
        int pivot = nums[left];
        int start = left;
        for (int i = left + 1; i <= right; i++) {
            if (nums[i] > pivot) {
                swap(nums[i], nums[left + 1]);//
                left++;
            }
        }
        swap(nums[left], nums[start]);
        if (k <= left - start ) {//
            quickSort(nums, start , left - 1, k, res); // 在左区间内
        }
        else {
            for (int i = start; i <= left; i++) {
                res.push_back(nums[i]);//存储结果
            }
            // k = left -start + 1时 即pivot为目标值
            if (k > left - start + 1) { // i ~ i+k : （k + 1）个数字 
                quickSort(nums, left + 1 , right, k - (left - start + 1), res);
            }
        }
    }
    int findKthLargest(vector<int>& nums, int k) {
        // 定义最小堆
        vector<int> ans;
        quickSort(nums, 0, nums.size() - 1, k, ans);
        return ans.back();
       
    }
};
```

### 56. 合并区间 [medium]
- 给出区间的集合，合并重叠的区间
- 先进行排序，然后进行遍历合并
  - 时间复杂度 O(NlogN) 空间复杂度
  - 使用运算符重载的方式定义二维数组的排序
```
class Solution {
public:
    struct cmp{
        bool operator()(vector<int>& a,  vector<int>& b) {
            return a[0] < b[0];
        }
    };
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(), cmp());
        vector<vector<int>> ans;
        if (intervals.empty())
            return ans;
        int l_min = intervals[0][0];
        int r_max = intervals[0][1];
        for (int i = 1; i < intervals.size(); i++) {
            if (r_max < intervals[i][0])
            {
                ans.push_back(vector<int>{l_min, r_max});
                r_max = intervals[i][1];
                l_min = intervals[i][0];
            }
            else {
                // 取当前对比的最大值
                r_max = max(intervals[i][1], r_max);
            }
        }
        ans.push_back(vector<int>{l_min, r_max});
        return ans;
    }
};
```

### 240. 搜索二维矩阵 [Medium]
- 二分查找法：
- 在对角线开始向下和向右的搜索

```
class Solution {
public:
    int rows = 0;
    int cols = 0;
    bool binarySearch(vector<vector<int>>& matrix, int target, int row) {
        //cout << "binarySearch" << endl;
        int left = row;
        int right = rows - 1;
        int col = row;
        while (left <= right) {
            //cout <<"l : " << left << " r :" << right << endl; 
            int mid = left + (right - left)/2;
            if (matrix[mid][col] == target) 
                return true;
            else if (matrix[mid][col] < target) {
                left = mid + 1;
            }
            else if (matrix[mid][col] > target) {
                right = mid - 1;
            }
        }
        int low = row;
        int high = cols - 1;
        while (low <= high) {
            //cout <<"low : " << low << " high :" << high << endl;
            int mid = low + (high - low)/2;
            if (matrix[row][mid] == target) 
                return true;
            else if (matrix[row][mid] < target) {
                low = mid + 1;
            }
            else if (matrix[row][mid] > target) {
                 high = mid - 1;
            }
        }
        return false;
    }
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        rows = matrix.size();
        cols = matrix[0].size();
        for (int i = 0; i < min(rows,cols); i ++) {
            if (binarySearch(matrix, target, i))
                return true;
        }
        return false;
    }
};
```
- **进阶做法**： 由于行和列都是有序的，可以将这个矩阵看作是一个搜索二叉树，
  - **将左下角元素作为根节点，向上都是小于根节点的元素，向右都是大于根节点的元素**
- 时间复杂度：`O(m+n)`
```
class Solution {
public:

    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int row = matrix.size() - 1;
        int col = 0;
        while(row >=0 && col < matrix[0].size()) {
            if (matrix[row][col] == target) 
                return true;
            else if (matrix[row][col] > target) {
                row--; // 向小的方向移动
            }
            else if (matrix[row][col] < target) {
                col++;
            }
        }
        return false;
    }
};
```