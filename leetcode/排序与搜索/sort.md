

- [排序算法](#排序算法)
  - [比较排序 v.s. 非比较排序](#比较排序-vs-非比较排序)
    - [0. 冒泡排序](#0-冒泡排序)
    - [1.快速排序](#1快速排序)
    - [2. 选择排序](#2-选择排序)
    - [3. 插入排序](#3-插入排序)
    - [4. 希尔排序 [nlogn]](#4-希尔排序-nlogn)
    - [5. 归并排序](#5-归并排序)
    - [6. 堆排序](#6-堆排序)
- [延伸题目](#延伸题目)
  - [56. 合并区间 [medium]](#56-合并区间-medium)
  - [75. 颜色分类 [Medium]](#75-颜色分类-medium)
  - [剑指21. 调整数组顺序使奇数位于偶数前面](#剑指21-调整数组顺序使奇数位于偶数前面)
  - [剑指40. 最小的k个数 [*]](#剑指40-最小的k个数-)
  - [215. 数组中的第K个最大元素 [Medium]*](#215-数组中的第k个最大元素-medium)
  - [347. 前K个高频元素 [Medium]*](#347-前k个高频元素-medium)
      - [基于快速排序的思想](#基于快速排序的思想)
      - [基于堆的思想](#基于堆的思想)
  - [386. 整数的字典序 [Medium] [字节 *]](#386-整数的字典序-medium-字节-)

排序算法
-------
面试中的基本盘，考察对基础/常用排序算法的理解和应用；也是面试八股中的重要内容。
**考察方面一**： 常见排序算法的概念思想、时间复杂度/空间复杂度，以及基本实现
**考察方面二**： 对部分排序算法的深入理解和应用： 快排的优化/快排与快速选择及堆排在选择topK/中位数问题
**发散延伸**： 字典序等




![](./summary.png)
动画图解：https://github.com/chefyuan/algorithm-base
https://zhuanlan.zhihu.com/p/60152722
*全表背诵*

------

### 比较排序 v.s. 非比较排序

常见的快速排序、归并排序、堆排序、冒泡排序等属于比较排序。在排序的最终结果里，元素之间的次序依赖于它们之间的比较。每个数都必须和其他数进行比较，才能确定自己的位置。

在冒泡排序之类的排序中，问题规模为n，又因为需要比较n次，所以平均时间复杂度为O(n²)。在归并排序、快速排序之类的排序中，问题规模通过分治法消减为logN次，所以时间复杂度平均O(nlogn)。
**比较排序的优势是，适用于各种规模的数据，也不在乎数据的分布，都能进行排序。可以说，比较排序适用于一切需要排序的情况。**

计数排序、基数排序、桶排序则属于非比较排序。非比较排序是通过确定每个元素之前，应该有多少个元素来排序。针对数组arr，计算arr[i]之前有多少个元素，则唯一确定了arr[i]在排序后数组中的位置。
**非比较排序只要确定每个元素之前的已有的元素个数即可，所有一次遍历即可解决。算法时间复杂度O(n)。非比较排序时间复杂度底，但由于非比较排序需要占用空间来确定唯一位置。所以对数据规模和数据分布有一定的要求。**



--------
#### 0. 冒泡排序
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

#### 1.快速排序
- 特点：左右交换，递归进行, **非稳定排序**
- 空间复杂度 O(LogN)
- 时间复杂度 O(NLogN)  最差情况下 O(N^2)
  - **时间复杂度与划分点选择相关，如果选中的划分点可以平分序列，那么总划分次数就会是logn**
  - **如果每次选中的划分点为最小/最大值，那么总有一个空区间，剩下的部分等价于线性处理，时间复杂度为O(n^2)**
  - 优化方法为：通常采用“三者值取中”方法，即比较r[low].key、r[high].key与r[(low+high)/2].key，**取三者中关键字为中值的元素为中间数** （解决序列部分有序的问题）

- 参考：https://blog.csdn.net/qq_19525389/article/details/81436838
- **快排优化**
  - 3种取基准的方法
    - 随机（rand函数）、固定（队首、队尾）、三数取中（队首、队中和队尾的中间数）
  - 4种优化方式：
      - 优化1：当待排序序列的长度分割到一定大小后，使用插入排序 **处理序列中有重复的情况** 当待排序列长度为5~20之间，此时使用插入排序能避免效率下降
      - 优化2：在一次分割结束后，可以把与Key相等的元素聚在一起，继续下次分割时，不用再对与key相等元素分割 **有效提高对有重复数据的序列处理速度** 将与当前pivot基准相同的元素在交换的时候都移到数组头部； 在完成交换后，再从头部交换到基准附近，从而将所有相等的值都聚合起来，大幅降低处理速度
      - 优化3：优化递归操作
      - 优化4：**使用并行或多线程处理子序列**

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


- **主元优化实例： 采用随机选取主元**

```
class Solution {
public:
    int partition(vector<int>& nums, int left, int right) {
        int start = left;
        int pivot = nums[start];
        // cout << left << " " << right << " " << pivot << endl;
   
        while (left < right) {
   
            while ((left < right) && (nums[right] >= pivot)) {
                right--;
            }
  
            while (left < right && nums[left] <= pivot) {
                left++;
            }
  
            if (left < right) {
                swap(nums[left], nums[right]);
            }
        }
        swap(nums[left], nums[start]);
        return left;
    }
    // ***** 关键
    int randomized_partition(vector<int>& nums, int left, int right) {
        int i = rand() % (right - left + 1) + left; // 随机选一个作为我们的主元
        swap(nums[left], nums[i]); // 交换随机元
        return partition(nums, left, right);
    }
    void quickSort(vector<int>& nums, int left, int right) {
        if (left >= right) {
            return;
        }
        int p = randomized_partition(nums, left, right);
        quickSort(nums, left, p - 1);
        quickSort(nums, p + 1, right);
    }
    vector<int> sortArray(vector<int>& nums) {
        quickSort(nums, 0, nums.size() - 1);
        return nums;
    }
};
```

- **代码精简版**

```
    void quick_sort(vector<int>& nums, int l, int r){
        if(l < r){
            int i = l, j = r;
            while(i < j){
                while(i < j && nums[j] >= nums[l]) --j;
                while(i < j && nums[i] <= nums[l]) ++i;
                swap(nums[i], nums[j]);
            }
            swap(nums[l], nums[i]);
            quick_sort(nums, l, i - 1);
            quick_sort(nums, i + 1, r);
        }
    }
```


#### 2. 选择排序

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


#### 3. 插入排序 
原理： 构建有序数组，**每次往前面构成的有序数组中插入**
- 时间复杂度： O(N^2)  最好的情况： O(N) 最差的情况： O(N^2)
- 空间复杂度:  O(1)
- 稳定性： **稳定 (不存在位置交换)**

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
#### 4. 希尔排序 [nlogn]
- 简单插入排序的升级
- 将整个数组按照gap分成不同的子集，每个子集进行插入排序，并逐步缩小gap值
- 时间复杂度比之前的O(N2)有缩小
  - 时间复杂度： O(NlogN)  最好的情况： O(N) 最差的情况： O(NlogN)
  - 空间复杂度:  O(1)
  - 不稳定
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
 
#### 5. 归并排序
- **基于分治思路**
- 建立在归并操作上的一种有效的排序算法。该算法是采用分治法（Divide and Conquer）的一个非常典型的应用。归并排序是一种稳定的排序方法。将已有序的子序列合并，得到完全有序的序列；即先使每个子序列有序，再使子序列段间有序。若将两个有序表合并成一个有序表，称为2-路归并
  - 时间复杂度： 稳定在`O(NLogN)` 空间复杂度 `O(N)`
  - 稳定排序
  - 适用场景：**数据量大，对稳定性有一定要求**


```
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> tmp;
    int i = left;
    int j = mid + 1;
    while(i <= mid && j <= right) {
        if (arr[i] > arr[j]) {
            tmp.push_back(arr[j]);
            j++;
        }
        else {
            tmp.push_back(arr[i]);
            i++;
        }
    }
    // 把没有遍历完的部分继续进行遍历
    while (i<=mid) {
        tmp.push_back(arr[i]);
        i++;
    }
    while (j <= right) {
        tmp.push_back(arr[j]);
        j++;
    }
    for (int i = 0; i <tmp.size(); i++) {
        arr[left++] = tmp[i];
    }
    cout << "after merge: "; 
    for (int i = 0; i < arr.size(); i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

}

void mergeSort(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}
```

#### 6. 堆排序 
- HeapSort
- 基于完全二叉树的结构：
  - 大顶堆 **[用于从小到大排序]** 根节点的值大于两个子节点的值；
  - 小顶堆 **[用于从大到小排序]** 根节点的值小于两个子节点的值
- 排序过程：
  - 1. 构建大顶堆 **自底而上地对节点进行调整**
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
- STL 的**priority_queue**更方便，优先队列的底层就是一个堆结构，在优先队列中，队首元素一定是当前队列中优先级最高的那一个。
- 通过 top() 函数来访问队首元素（也可称为堆顶元素），也就是优先级最高的元素。
- push()插入元素

```
priority_queue< int, vector<int>, greater<int> > q;  // 小顶堆
priority_queue< int, vector<int>, less<int> > q;     // 大顶堆
```

--------

## 延伸题目

### 56. 合并区间 [medium]
> 数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间

- 问题并不复杂，模拟整个合并区间的过程即可
- 完整思路： 
  - 给出区间的集合，合并重叠的区间
  - 先进行排序，然后进行遍历合并
    - 时间复杂度 O(NlogN) 空间复杂度
    - 使用运算符重载的方式定义二维数组的排序
  - 遍历合并时主要时判断前后两个区间的范围，动态更新最右侧的阈值即可
- 时间复杂度 O(nlogn) 主要耗费在排序上
  - **需要掌握基于重载运算符方式的排序实现**
  
```
class Solution {
public:
    struct cmp{
        bool operator()(vector<int>& a,  vector<int>& b) {
            return a[0] < b[0]; // **升序排序  如果是 > 则是用于降序**
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
### 75. 颜色分类 [Medium]
- 三种颜色 0 1 2 ， 要实现对这些颜色的原地排序 从小到大排列


- 双指针法： **left指针控制0， right指针控制2，进行交换，只需要遍历一次即可完成排序**
  - 遇到0，与left进行交换，即往左甩
  - 遇到2，则与right进行交换，往右甩
  - `left < right`； 
    - 技巧点： 当进行right交换时，交换完成后要回退一步，保证避免遗漏对元素的处理，*可能将2又换到当前位置*
- 时间复杂度： `O(N)`

- 关键点： **双指针** **一次遍历**

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
        }
    }
};
```

### 剑指21. 调整数组顺序使奇数位于偶数前面
> 使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

- 跟LC75题思路相似，原地移动即可，设置left指针，将所有奇数向left指针位置交换即可
- 时间复杂度 O(N)
```
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        int left = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] % 2 ) {
                swap(nums[left], nums[i]);
                left++;
            }
        }
        return nums;
    }
};
```


### 剑指40. 最小的k个数 [*]
> 输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
- 与下面的两道题基本相似
- 通过对堆排序或者/对快排进行改造，减少处理逻辑，来获得高效的算法

- 堆排序/快速排序
  - 堆排序：维护大顶堆，更新堆顶保持一个最小k个数字
  - 快速选择： 维护左侧部分小于/等于划分点

- 堆排序
```
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        priority_queue<int, vector<int>, less<int>> p;

        for (int i = 0; i < arr.size(); i++) {
            if (i < k) {
                p.push(arr[i]);
            }
            else {
                if (!p.empty() && arr[i] < p.top()) {
                    p.pop();
                    p.push(arr[i]);
                }
            }
        } 
        vector<int> ans;
        while (!p.empty()) {
            ans.push_back(p.top());
            p.pop();
        }
        return ans;
    }
```
- 手写大顶堆 时间复杂度O(NlogK) 空间复杂度O(k)

```
class Solution {
public:
    void adjustHeap(vector<int>& arr, int len, int index) {
        int maxIdx = index;
        if (index*2 + 1 < len && arr[index*2 + 1] > arr[maxIdx]) maxIdx = 2*index + 1;
        if (index*2 + 2 < len && arr[index*2 + 2] > arr[maxIdx]) maxIdx = 2*index + 2;
        if (index != maxIdx) {
            swap(arr[index], arr[maxIdx]);
            adjustHeap(arr, len, maxIdx);
        }
    }
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        //priority_queue<int, vector<int>, less<int>> p;
        if (k >= arr.size())
            return arr;
        // 构建大顶堆
        for (int i = k/2 - 1; i >= 0; i--) {
            adjustHeap(arr, k, i);
        }
        //cout << arr[0] << arr[1] << arr[2] << endl;
        for (int i = k; i < arr.size(); i++) {
            if (arr[i] <= arr[0]) {
                swap(arr[i], arr[0]);
                adjustHeap(arr, k, 0);
            }
        } 
        vector<int> ans;
        return vector<int>(arr.begin(), arr.begin() + k);
    }
};
```
- **快速排序的改进： 快速选择**
  - 时间复杂度O(N)
  - 只关注于找到比当前划分点小于/等于的元素。
  - 并在目标范围内进行递归，不会同时左右都递归，大幅降低时间复杂度
    - 每次都判断第k个元素的区间位置(从小到大的第k个)
```
class Solution {
public:
    
    void quickSelect(vector<int>& arr, int left, int right, int k,vector<int>& ans ) {
        int start = left;
        int pivot = arr[left];
        for (int i = left + 1; i <= right; i++) {
            if (arr[i] <= pivot) { //取最小的部分
                swap(arr[left + 1], arr[i]);
                left++; // 关键
            }
        }
        swap(arr[left], arr[start]);
        
        //cout << left << start << " " << k << endl;
        // 方向判断 关键
        if (k <= (left - start)) {
            quickSelect(arr, start, left - 1, k, ans);
        }
        else {
            for (int i = start; i <= left; i++) {
                ans.push_back(arr[i]);
            }
            if (k > (left - start + 1)) {
                quickSelect(arr, left + 1, right, k - (left - start + 1), ans);
            }
        }
    }
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        //priority_queue<int, vector<int>, less<int>> p;
        if (k >= arr.size())
            return arr;
        
        vector<int> ans;
        if (k == 0)
            return ans;
        quickSelect(arr, 0, arr.size() - 1, k, ans);
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
- 基于快速排序的思路 **快速选择**
  - 与上一题思路基本一致，只要在指定区域进行搜索即可
  - 时间复杂度: O(N) 空间复杂度:`O(logN)` 递归栈的深度

```
class Solution {
public:
    void quickSort(vector<int>& nums, int left, int right, int k, vector<int>& res) {
        int pivot = nums[left];
        int start = left;
        for (int i = left + 1; i <= right; i++) {
            if (nums[i] > pivot) { // 从大到小排序
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

- **自定义堆结构，构建小顶堆，取堆顶作为目标值**
    - s1：递归建堆
    - s2: 对于堆外元素，当其大于堆顶元素时进行堆调整
    - s3 返回堆顶
```
class Solution {
public:
    void adjustHeap(vector<int>& nums, int length, int index) {
        int minIdx = index;
        if (2*index + 1 < length && nums[2*index + 1] < nums[minIdx])
            minIdx = 2*index + 1;
        if (2*index + 2 < length && nums[2*index + 2] < nums[minIdx])
            minIdx = 2*index + 2;
        // 调整子树
        if (minIdx != index) {
            swap(nums[index], nums[minIdx]);
            adjustHeap(nums,length,minIdx); 
        }
    }
    int findKthLargest(vector<int>& nums, int k) {
        for (int i = k/2 - 1; i >=0; i--) {
            adjustHeap(nums, k, i);
        }
        for (int i = k; i < nums.size(); i++) {
            if (nums[i] > nums[0]) {
                nums[0] = nums[i];
                adjustHeap(nums,k,0);
            }
        }
        return nums[0];
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

### 386. 整数的字典序 [Medium] [字节 *]
> 给定一个整数 n, 返回从 1 到 n 的字典顺序。
```
例如，
给定 n =13，返回 [1,10,11,12,13,2,3,4,5,6,7,8,9]
```

- **字典序可以视为树结构**
- 首个数字不能是0，但其他位置可以是0，
  - 通过DFS对树结构进行遍历，保存结果
  - **自顶而下**的遍历方式，每次都进行结果存储
- 时间复杂度 O(N) 
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