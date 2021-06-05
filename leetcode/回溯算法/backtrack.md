- [回溯算法](#回溯算法)
  - [基本思想](#基本思想)
  - [套路框架](#套路框架)
  - [17. 电话号码的字母组合 [Medium]](#17-电话号码的字母组合-medium)
  - [22. 括号生成 [Medium]](#22-括号生成-medium)
  - [39. 组合数 [Medium] [ByteDance]](#39-组合数-medium-bytedance)
  - [40. 组合数 II](#40-组合数-ii)
  - [46. 全排列 Permutations [MEDIUM]](#46-全排列-permutations-medium)
  - [47. 全排列 II](#47-全排列-ii)
  - [51. N皇后 I [HARD]](#51-n皇后-i-hard)
  - [52. N皇后 2 [HARD]](#52-n皇后-2-hard)
  - [x. N皇后思考](#x-n皇后思考)
  - [78. 子集 [Medium]](#78-子集-medium)
  - [79. 单词搜索 [Medium]](#79-单词搜索-medium)
  - [93. 复原IP地址 [美团]](#93-复原ip地址-美团)
  - [679. 24点游戏 *](#679-24点游戏-)
  - [698. 划分为k个相等的子集 *](#698-划分为k个相等的子集-)

## 回溯算法

### 基本思想
**穷举**状态，遍历决策空间，选择合适的结果

### 套路框架

```
result = []
def backtrack (路径，选择列表)：
    if 满足条件：
        result.add（路径）
        return

    for choice  in  选择列表：
        make choice
        backtrack(路径， 选择列表)
        撤销选择
```



*****


### 17. 电话号码的字母组合 [Medium]
- 回溯法解题的思路总体属于中等偏上的难度
- 跟前面分析一致，需要先确定选择列表和路径列表，以及make choice的规则
- 时间复杂度： O（N!） 空间复杂度O(N)
- 代码更新：
```c++
class Solution {
public:
    unordered_map<char, string> numDigits;
    vector<string> ans;
    void backTrack(string digits, int index, string& path) {
        if (path.size() == digits.size()) {
            ans.push_back(path);
            return;
        }
        string s = numDigits[digits[index]];
        for (int i = 0; i < s.size(); i++) {
            path.push_back(s[i]);
            backTrack(digits, index + 1, path);
            path.pop_back();
        }
    }
    vector<string> letterCombinations(string digits) {
        if (digits.empty()) return ans;
        int beg = 0;
        for (int i = 2; i <= 9; i++) {
            int l = (i == 8 or i < 7) ? 3 :4;
            string tmp;
            for (int j = 0; j < l; j++) {
                tmp.push_back('a' + beg);
                beg++;
            }
            numDigits['0' + i] = tmp;
        }
        string path;
        backTrack(digits, 0, path);
        return ans;
    }
};
```

```c++
class Solution {
public:
    unordered_map<char, string> numDigits;
    string digitsTarget;
    vector<string> ans;
    vector<string> letterCombinations(string digits) {
        if (digits.empty()) 
            return ans;
        numDigits.insert({'2',"abc"});
        numDigits.insert({'3',"def"});
        numDigits.insert({'4',"ghi"});
        numDigits.insert({'5',"jkl"});
        numDigits.insert({'6',"mno"});
        numDigits.insert({'7',"pqrs"});
        numDigits.insert({'8',"tuv"});
        numDigits.insert({'9',"wxyz"});
        digitsTarget = digits;
        string path = "";
        backTrack(path,0);
        return ans;
    }
    void backTrack(string& path, int num) {

        if (path.size() == digitsTarget.size()) {
            ans.push_back(path);
            return;
        }
        string digitCur = numDigits[digitsTarget[num]];
        for (auto s: digitCur) {
             path.push_back(s);
             backTrack(path, num + 1);
             path.pop_back();
        }
    }
};
```

### 22. 括号生成 [Medium]
> 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。


- 针对左右括号的依赖，在进行回溯时添加依赖
  - 当`left < n`时才，才能添加左括号
  - 当`right < left` 时，才能添加右括号

```c++
class Solution {
public:
    int total = 0;
    vector<string> ans;
    vector<string> generateParenthesis(int n) {
        if (!n)
            return ans;
        total = n*2;
       // cout << total <<endl;
        string path = "";
        backTrack(path, 0, 0, 0);
        return ans;
    }
    void backTrack(string& path, int step, int right, int left) {
        if (step == total) {
            ans.push_back(path);
           // cout << "return" << endl;
            return;
        }
        //cout << step << endl;
        // 放置(的条件
        if (left < total/2) {
            path.push_back('(');
            backTrack(path, step + 1, right, left + 1);
            path.pop_back();
        }
        // 放）的条件
        if (right < left) {
            path.push_back(')');
            backTrack(path, step + 1, right + 1, left);
            path.pop_back();
        }
    }
};
```

- 更新代码写法：

```c++
class Solution {
public:
    vector<string> ans;
    void backTrack(string& path, int left, int right, int n) {
        if (path.size() == n) {
            ans.push_back(path);
            return;
        }
        if (left < n/2) {
            path.push_back('(');
            backTrack(path, left + 1, right, n);
            path.pop_back();
        }
        if (right < left) {
            path.push_back(')');
            backTrack(path, left, right + 1, n);
            path.pop_back();
        }
    }
    vector<string> generateParenthesis(int n) {
        string s;
        backTrack(s, 0, 0, n * 2);
        return ans;
    }
};
```

### 39. 组合数 [Medium] [ByteDance]
> 给定一个**无重复元素的数组** candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合
- 所有数字（包括 target）都是**正整数。 解集不能包含重复的组合**。
- 解题关键：
  - 循环结束的条件： **由于都是正整数，所以当curSum值大于target时必然要结束**
  - 如何避免重复查找： **每次仅寻找大于等于当前组合数的候选值**
  - 时间复杂度 O(N!) 空间复杂度O(N) 

```c++
#include<algorithm>
class Solution {
public:
    vector<vector<int>> res;
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        if (candidates.empty())
            return res;
        vector<int> path;
        backTrack(candidates, path, target, 0);
        return res;
    }
    void backTrack(vector<int>& candidates, vector<int>& path, int target, int curSum) {
        if (curSum > target) {
            return;
        }
        if (curSum == target) {
            res.push_back(path);
            return;
        }
        for (int i = 0; i < candidates.size(); i++) {
            if (!path.empty()) {
                // 去重处理
                if (path.back() > candidates[i])
                    continue;
            }
            path.push_back(candidates[i]);
            backTrack(candidates, path, target, curSum + candidates[i]);
            path.pop_back();
        }
        return;
    }
};
```

### 40. 组合数 II
> 给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的每个数字在每个组合中只能使用一次

- 与LC39的唯一区别在于，这个**数组是存在重复元素的**, **每个元素只能使用一次**
  - 首先进行排序
  - 在之前写法的基础上，**增加一个重复判断**： 当遍历时出现连续值时进行跳过，避免重复值的二次利用
  - **结合遍历开始的index**和重复判断逻辑，完成最终的回溯

- 关键点： **遍历起点**  **重复判断**

```c++
class Solution {
public:
    vector<vector<int>> ans;
    void backTrack(vector<int>& candidates, int target, int begin, int sum,vector<int>& path) {
        if (sum == target) {
            vector<int> tmp;
            for (auto p : path) tmp.push_back(candidates[p]);
            ans.push_back(tmp);
            return;  
        }
        if (sum > target)
            return;
        for (int i = begin; i < candidates.size(); i++) {
            if (i > begin && candidates[i] == candidates[i - 1]) {
                continue;
            }
            if (path.empty() || path.back() < i) {
                path.push_back(i);
                backTrack(candidates, target, i + 1, sum + candidates[i], path);
                path.pop_back();
            }
        }
    }
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<int> path;
        backTrack(candidates, target,0, 0, path);
        return ans;
    }
};
```

### 46. 全排列 Permutations [MEDIUM]
> 给定不含重复元素的数组，求所有全排列的结果


- 模板的标准使用，指定path和状态集合(即选择集合)
- 需要判断当前选择的path是否有重复
- 时间复杂度 `O(N!)` 空间复杂度`O(N)` **取决与栈的深度，为N**
```c++
#include<algorithm>
class Solution {
public:
    vector<vector<int>> res;
    vector<vector<int>> permute(vector<int>& nums) {
        vector<int> path;
        backtrack(path, nums);
        return res;
    }
    void backtrack(vector<int> path, vector<int> nums) {
        if (path.size() == nums.size()) {
            res.push_back(path);
            return;
        }
        for (int i = 0; i < nums.size(); i ++) {
            if (find(path.begin(), path.end(), nums[i]) != path.end())
                continue;
            path.push_back(nums[i]);
            backtrack(path, nums);
            path.pop_back();
        }
    }
};
```
### 47. 全排列 II 
> 给定一个**可包含重复数字**的序列 nums ，按任意顺序 返回所有不重复的全排列。

- 与LC46做法一致，主要在于此处存储重复元素
- 如何**避免出现重复的排列结果**是解题的关键点
- 对于 `1 1 2`关键点就在于对于重复的1，**只能允许一种遍历顺序**， 否则结果就会重复
  - 因此，在回溯中添加对重复元素的遍历控制： **通过限制前一个相同元素是否被遍历过**，从而控制了遍历顺序
```
if (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1])
                continue;
```
或者
```
if (i > 0 && nums[i] == nums[i - 1] && visited[i - 1])
                continue;
```
- 关键点； **`回溯模版`**  **`重复值控制`**
```
class Solution {
public:
    vector<vector<int>> ans;
    void backTrack(vector<int> nums, vector<int>& path, int * visited) {
        if (path.size() == nums.size()) {
            ans.push_back(path);
            return;
        }
        for (int i = 0; i < nums.size(); i++) {
            // 重复处理
            if (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1])
                continue;
            if (!visited[i]) {
                path.push_back(nums[i]);
                visited[i] = 1;
                backTrack(nums, path, visited);
                path.pop_back();
                visited[i] = 0;
            }
        }
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        int visited[10] = {0};
        vector<int> path;
        sort(nums.begin(), nums.end());
        backTrack(nums, path, visited);
        return ans;
    }
};
```

### 51. N皇后 I [HARD]

- 选择列表：cols即多少列；每次逐行遍历，每次做选择时验证选择是否符合N皇后冲突规则
  - 验证： 当前位置是否与之前的同列位置/左上线/右上线存在Q **遍历非常耗时，可以通过哈希表为每列/线设置Q标识，以提高速度**
  - 选择不冲突的填入board
  - 选择完成后继续向下遍历；然后撤销选择，即修改board数据，进行回溯
- 路径列表：即当前的表盘board
- 时间复杂度 `O(N!)` 空间复杂度`O(N)` 取决于栈的深度(N行)

```
class Solution {
public:
    vector<vector<string>> res;
    int rows = 0;
    vector<vector<string>> solveNQueens(int n) {
        vector<string> board;
        string row;
        for (int i = 0; i < n; i++) {
            row.push_back('.');
        }
        for (int i = 0; i < n; i++) {
            board.push_back(row);
        }
        backTrack(board, 0);
        return res;
    }
    void backTrack(vector<string>& board, int row) {
        
        if (row == board.size()) {
            res.push_back(board);
            return; 
        }
        int cols = board[row].size();
        for (int col = 0; col < cols; col++) {
            // make choice ; 
            // 判断当前是否能放
            if (!isValid(row,col,board)) {
                continue;
            }
            // 添加到path中
            board[row][col] = 'Q';
            backTrack(board, row + 1);
            // 撤销选择
            board[row][col] = '.';
        }
    }
    bool isValid(int row, int col,vector<string> board) {
        int cols = board[row].size();
        // 同列上是否有Q
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q')
                return false;
        }
        // 同行不用考虑
        // 右上方是否有Q： 斜线方向
        for (int i = row - 1, j = col + 1; i >= 0 && j < cols; i-- , j++) {
            if (board[i][j] == 'Q')
                return false;
        }
        // 左上方是否有Q： 斜线方向
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i-- , j--) {
            if (board[i][j] == 'Q')
                return false;
        }
        return true;
    }
};
```
- **优化：** 基于set进行验证优化,设置三个set分别对应各列/线中有Q的索引
```
class Solution {
public:
    vector<vector<string>> res;
    unordered_set<int> setCols;
    unordered_set<int> setDiag1;
    unordered_set<int> setDiag2;

    int rows = 0;
    vector<vector<string>> solveNQueens(int n) {
        vector<string> board;
        string row;
        for (int i = 0; i < n; i++) {
            row.push_back('.');
        }
        for (int i = 0; i < n; i++) {
            board.push_back(row);
        }
        backTrack(board, 0);
        return res;
    }
    void backTrack(vector<string>& board, int row) {
        
        if (row == board.size()) {
            res.push_back(board);
            return; 
        }
        int cols = board[row].size();
        for (int col = 0; col < cols; col++) {
            // make choice ; 
            // 判断当前是否能放
            // 同列冲突
            if (setCols.find(col) != setCols.end()) continue;
            if (setDiag1.find(row - col) != setDiag1.end()) continue;
            if (setDiag2.find(row + col) != setDiag2.end()) continue;

            // 添加到path中
            board[row][col] = 'Q';
            setCols.insert(col);
            setDiag1.insert(row - col);
            setDiag2.insert(row + col);
            backTrack(board, row + 1);
            // 撤销选择
            board[row][col] = '.';
            setCols.erase(col);
            setDiag1.erase(row - col);
            setDiag2.erase(row + col);
        }
    }
};
```
### 52. N皇后 2 [HARD]
- 只需要求出答案数量
- 省去存储表盘board的过程，只需要保留哈希表对结果进行验证

```
class Solution {
public:
    int totalNQueens(int n) {
        unordered_set<int> setCols;
        unordered_set<int> diag1;
        unordered_set<int> diag2;
        return backTrack(0, n, setCols, diag1, diag2);
    }
    int backTrack(int row, int n, unordered_set<int>& setCols, unordered_set<int>& diag1, unordered_set<int>& diag2) {
        if (n == row) {
            return 1;
        }
        int count = 0;
        for (int col = 0; col < n; col++) {
            if (setCols.find(col) != setCols.end()) continue;
            if (diag1.find(row + col) != diag1.end()) continue;
            if (diag2.find(row - col) != diag2.end()) continue;
            setCols.insert(col);
            diag1.insert(row + col);
            diag2.insert(row - col);
            count += backTrack(row + 1, n, setCols, diag1, diag2);
            setCols.erase(col);
            diag1.erase(row + col);
            diag2.erase(row - col);
        }
        return count;
    }
};
```
### x. N皇后思考
- 从某种程度上将，回溯法就是dp的暴力版本，穷举所有可能的情况
- 时间复杂度`O(N!)` 空间复杂度O(N)
- 当我们在遍历时如果只需要拿到一个合理结果返回即可，那么就可以修改上面的代码，当第一次遍历到底即进行返回

**调整返回逻辑**
```
   bool backTrack(vector<string>& board, int row) {
        if (row == board.size()) {
            res.push_back(board);
            return true;  
        }
        int cols = board[row].size();
        for (int col = 0; col < cols; col++) {
            // make choice ; 
            // 判断当前是否能放
            // 同列冲突
            if (setCols.find(col) != setCols.end()) continue;
            if (setDiag1.find(row - col) != setDiag1.end()) continue;
            if (setDiag2.find(row + col) != setDiag2.end()) continue;

            // 添加到path中
            board[row][col] = 'Q';
            setCols.insert(col);
            setDiag1.insert(row - col);
            setDiag2.insert(row + col);
            // 调整返回逻辑

            if (backTrack(board, row + 1))
                return true;
            // 撤销选择
            board[row][col] = '.';
            setCols.erase(col);
            setDiag1.erase(row - col);
            setDiag2.erase(row + col);
        }
        return false;
    }
};
```


### 78. 子集 [Medium]
- 计算数组的全部子集，要避免重复子集的出现
- 本质上就是穷举，需要添加重复子集的处理 **相当于添加剪枝操作**
  - 由于数组有序，只需要保证子集内同样有序，每次仅添加大于当前最大值的元素进入选择列表中即可 
  - 由于每层都可以算是子集，所以每次回溯都能得到一个结果，需要每次都把结果添加到结果list中


```
class Solution {
public:
    vector<vector<int>> ans;
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<int> subset;
        backTrack(subset, nums);
        return ans;
    }
    void backTrack(vector<int> subset, vector<int> nums) {
        ans.push_back(subset);
        if (subset.size() == nums.size()) {
            return;
        }
        for (int i = 0; i < nums.size(); i++) {
            if (subset.empty()) {
                subset.push_back(nums[i]);
                backTrack(subset, nums);
                subset.pop_back();
            }
            else {
                if (nums[i] > subset.back()) {
                    subset.push_back(nums[i]);
                    backTrack(subset, nums);
                    subset.pop_back();
                }
            }

        }
    }
};
```

### 79. 单词搜索 [Medium]
- 需要借助辅助数组避免字母的重复使用，即剪枝
- 首字母通过外层迭代循环来找到，然后进入回溯寻找当前位置下的结果
- 时间复杂度 `O(MN*3^L)` 空间复杂度 `O(min(L,MN))` L : word长度


```
class Solution {
public:
    int direct[4][2] = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
    vector<vector<char>> visited;
    bool exist(vector<vector<char>>& board, string word) {
        if (word.empty() || board.empty()) {
            return false;
        }
        visited.assign(board.begin(), board.end());
        int row = 0;
        int col = 0;
        string cur;
        cur.push_back(word[0]);
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); j++) {
                if (board[i][j] == word[0]) {
                    visited[i][j] = '0';
                    if (backTrack(board, cur, word, 1, i, j))
                        return true;
                    visited[i][j] = '1';
                }
            }
        }
        return false;
    }
    bool backTrack(vector<vector<char>>& board, string& cur, string& target, int index, int row, int col) {
        if (cur.size() == target.size()) {
            return true;
        }
        for (int i = 0; i < 4; i++) {
            int _row = row + direct[i][0];
            int _col = col + direct[i][1];
            if (_row >=0 &&  _row < board.size() &&
                _col >=0 &&  _col < board[0].size()) {
                    if (board[_row][_col] == target[index] && visited[_row][_col] !='0') {
                        cur.push_back(board[_row][_col]);
                        visited[_row][_col] = '0';
                        bool flag = backTrack(board, cur, target, index + 1, _row, _col);
                        if (flag)
                            return true;
                        cur.pop_back();
                        visited[_row][_col] = '1';
                    }
                }
            }
        return false;
    }
};
```
### 93. 复原IP地址 [美团]
> 给定一个只包含数字的字符串，用以表示一个 IP 地址，返回所有可能从 s 获得的 有效 IP 地址 。你可以按任何顺序返回答案。
有效 IP 地址 正好由四个整数（**每个整数位于 0 到 255 之间组成，且不能含有前导 0**)


- 回溯法解题
- 需要考虑比较多的限制：
  - 所有字符都要用上，回溯才能结束
  - 每段数字要合法， 当出现非法数字段时就要停止回溯循环
  - 对于0数字，如果出现在前导位置（即某段的第一个位置）那么该段只能是0
- 时间复杂度： $3^{SEG_COUNT}×∣s∣$

```c++
class Solution {
public:
    vector<string> ans;
    void backTrack(string& s, int segId, int segStart, vector<string>& paths) {
        if (segId == 4) {
            // 临界情况判断： 当所有字符都用上才可以
            if (segStart == s.size()) {
                string tmp = "";
                for (auto p : paths) {
                    if (tmp.empty())
                        tmp += p;
                    else {
                        tmp += ".";
                        tmp += p;
                    }
                }
                ans.push_back(tmp);   
            }
            return;
        }
        // 当已经到达尾部
        if (segStart >= s.size()) {
            return;
        }
        // 如果前导为0的话，当前段只能为0
        if (s[segStart] == '0') {
            paths.push_back("0");
            backTrack(s, segId + 1, segStart + 1, paths);
            paths.pop_back();
        }
        // 标准的回溯， 需要判断是否在合法区间，超出合法区间就停止
        int num = 0;
        for (int i = segStart; i < s.size(); i++) {
            num = num * 10 + (s[i] - '0');
            if (num <= 255 && num > 0) {
                paths.push_back(to_string(num));
                backTrack(s, segId + 1, i + 1, paths);
                paths.pop_back();
            } 
            else {
                break;
            }
        }
    }
    vector<string> restoreIpAddresses(string s) {
        vector<string> path;
        backTrack(s, 0, 0, path);
        return ans;
    }
};
```
### 679. 24点游戏 *
> 有 4 张写有 1 到 9 数字的牌。你需要判断是否能通过 *，/，+，-，(，) 的运算得到 24

- 4张牌，使用3次运算组合， **通过回溯法来完成这一搜索**
- 构建运算的过程为： 
  - 从4张牌中选择两张来计算， 计算结果加入剩下的两张中，构成3张
  - 递归地从剩下的牌中选择2张计算，更新到剩下的结果，构成2个值
  - 最后再选择一个运算，**得到运算结果** 即当数组只有一个元素时进行值判断和返回
  - 对应的代码为：
  ```c++
    vector<double> res;
    for (int k = 0; k < s; k++) {
        // 仅保留未使用过的数字
        if (k != i && k != j)
            res.push_back(tmp[k]);
    }
  ```
- 在计算中，要考虑到减法和除法不满足交换规律，因此总共可以有6种计算方式， 通过添加限制来跳过对加法和乘法的重复处理:
```c++
for (int k = 0; k < 4; k++) {
    // 对于加法和乘法 只进行一遍计算， 因为满足交换律
    if (k < 2 && i > j) {
        continue;
    }
    //
}
```
- 关键点： **`四则运算处理`**   **`基于临时数组进行抽卡`**
 
```c++
class Solution {
public:
    bool backTrack(vector<double>& tmp) {
        if (tmp.size() == 0) return false;
        if (tmp.size() == 1) {
            return fabs(tmp[0] - 24) < 1e-6;
        }
        int s = tmp.size();
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                if (i != j) {
                    vector<double> res;
                    for (int k = 0; k < s; k++) {
                        // 仅保留未使用过的数字
                        if (k != i && k != j)
                            res.push_back(tmp[k]);
                    }
                    for (int k = 0; k < 4; k++) {
                        // 对于加法和乘法 只进行一遍计算， 因为满足交换律
                        if (k < 2 && i > j) {
                            continue;
                        }
                        if (k == 0) {
                            res.push_back(tmp[i] + tmp[j]);
                        }
                        else if (k == 1) {
                            res.push_back(tmp[i] * tmp[j]);
                        }
                        else if (k == 2) {
                            res.push_back(tmp[i] - tmp[j]);
                        }
                        else {
                            if (fabs(tmp[j]) > 1e-6)
                                res.push_back(tmp[i] / tmp[j]);
                            else {
                                continue;
                            }
                        }
                        if (backTrack(res)) {
                            return true;
                        }
                        res.pop_back();
                    }

                }
            }
        }
        return false;
    }
    bool judgePoint24(vector<int>& cards) {
        vector<double> res;
        for (auto p : cards) {
            res.push_back((double) p);
        }
        return backTrack(res);
    }
};
```



### 698. 划分为k个相等的子集 *
> 给定一个整数数组  nums 和一个正整数 k，找出是否有可能把这个数组分成 k 个非空子集，其总和都相等

```
输入： nums = [4, 3, 2, 3, 5, 2, 1], k = 4
输出： True
说明： 有可能将其分成 4 个子集（5），（1,4），（2,3），（2,3）等于总和。
```

- 总体上看可以作为回溯题目来做，将数组分配到k个不同桶里
  - 回溯时，尝试当前值在不同桶的分配
  - 同时进行剪枝，删除超过目标值的情况
  - 并提前对数组进行**降序排序**，那么先入桶的即为最大的值，可以减少后面的搜索过程
    - 此外有一些特殊情况，可能存在超时，需要进行特别的剪枝操作： *具体逻辑还没理解*
    - 即对于当前入桶值如果之前已经出现过，则直接跳过，相当于避免同一情况重复被不同桶遍历

```c++
class Solution {
public:
    int visited[100] = {0};
    bool backTrack(vector<int>& nums, int target, int index, vector<int>& sum, int k) {
        if (index == nums.size()) return true;
        for (int i = 0; i < k; i++) {
            int tmp = nums[index] + sum[i];
            if (tmp > target)
                continue;
            // 避免超时操作， 避免重复添加
            int j = 0;
            while (j < i && sum[j] != tmp)
                j++;
            if (j < i) continue;
            // 
            sum[i] = tmp;
            if (backTrack(nums, target, index + 1, sum, k))
                return true;
            sum[i] -=  nums[index];
        }
        return false;
    }
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        if (nums.empty()) return false;
        int sum = 0;
        int numMax = INT_MIN;
        int numMin = INT_MAX;
        for (auto p : nums) {
            sum += p;
            numMin = min(numMin, p);
            numMax = max(p, numMax);
        }
        int target = sum / k;
        if (sum % k || target < numMax) return false;
        cout << target << endl;
        if (numMax != target && numMin != target && (numMax + numMin) > target)
            return false;
        sort(nums.begin(), nums.end(), greater<int>());
        vector<int> s (k);
        return backTrack(nums, target, 0, s, k);

    }
};

```