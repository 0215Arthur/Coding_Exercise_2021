
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