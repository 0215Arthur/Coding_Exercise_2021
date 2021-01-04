
### 384. 数组打乱shuffle

- 两种思路： 暴力算法：每次从数组中随机选择一个数字处理，重复N次构成新数组，并移除原数组的相应元素(即保证无放回随机采样，采样空间为n!)
    - 时间复杂度：O(n^2) 
- Fisher-Yates 洗牌算法：每次迭代中，生成一个范围在当前下标到数组末尾元素下标之间的随机整数
    - 时间复杂度: O(n) 使用rand()%()来生成索引，不需要额外的空间；
    - 理论上同样是n!的采样空间

```class Solution {
private:
       vector<int> data;
public:
    Solution(vector<int>& nums) {
        //vector<int>* array=nums;
        data=nums;
    }
    
    /** Resets the array to its original configuration and return it. */
    vector<int> reset() {

    /** Returns a random shuffling of the array. */
    vector<int> shuffle() {
        vector<int> nums(data);
        for(int i=0;i<nums.size();i++){
            cout<<rand()%(nums.size()-i)+i<<endl;
            swap(nums[i],nums[rand()%(nums.size()-i)+i]);
        }
        return nums;
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(nums);
 * vector<int> param_1 = obj->reset();
 * vector<int> param_2 = obj->shuffle();
 */
```

### 155. 最小栈

- 问题：复现栈的基本功能，并能取得当前栈的最小值
- 额外使用一个辅助栈**记录每步的最小值**，空间复杂度为O(n)，时间复杂度同样为O(n)

```
class MinStack {
    stack<int> x_stack;
    stack<int> min_stack;
public:
    /** initialize your data structure here. */
    MinStack() {
        min_stack.push(INT_MAX);
    }
    
    void push(int x) {
        x_stack.push(x);
        min_stack.push(min(min_stack.top(),x));
    }
    
    void pop() {
        x_stack.pop();
        min_stack.pop();
    }
    
    int top() {
        return x_stack.top();
    }
    
    int getMin() {
        return min_stack.top();
    }
};

```
- 优化：把空间复杂度降低到O(1)
- 在stack中直接存储元素差值

```
class MinStack {
    stack<long> x_stack;
    long min_value;
public:
    /** initialize your data structure here. */
    MinStack() {
        min_value=0;
        
    }
    
    void push(int x) {
        if(x_stack.empty()){
            min_value=x;
            x_stack.push(0);
        }
        else{
            long diff=x-min_value;
            if(diff<0){
                x_stack.push(diff);
                min_value=x;
            }
            else{
                x_stack.push(diff);
            }

        }
        
    }
    
    void pop() {
        long diff=x_stack.top();
        x_stack.pop();
        min_value=diff<0?min_value-diff:min_value;
        //return diff<0?min_value:min_value+diff;
    }
    int top() {
        long diff=x_stack.top();
        return diff<0?min_value:min_value+diff;
        //return x_stack.top();
    }
    
    int getMin() {
        return min_value;
    }
};
```