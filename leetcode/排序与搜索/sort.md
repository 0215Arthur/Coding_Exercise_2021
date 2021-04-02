
![](./summary.png)
动画图解：https://github.com/chefyuan/algorithm-base
[TOC]

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

### 插入排序

### 希尔排序



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