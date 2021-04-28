
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

### 412. Fizz Buzz
- 简单题，普通的条件判断即可完成
- 当目标值增多时，需要写比较多的条件判断，这时需要进行简单优化，省去大量条件判断语句

- 基本写法： 三次判断
```class Solution {
public:
    vector<string> fizzBuzz(int n) {
        vector<string> res;
        for(int i=1;i<=n;i++){
            if(i%3==0 && i%5==0){
                res.push_back("FizzBuzz");
            }
            else if(i%3==0){
                res.push_back("Fizz");
            }
            else if(i%5==0){
                res.push_back("Buzz");
            }
            else{
                res.push_back(to_string(i));
            }
        }
        return res;
   }
};
```

- 优化写法：**利用字符串来记录整除的结果，总共只需两次条件判断**： 提前将映射存在map中，先对关键值进行计算，得到判断结果

```class Solution {
public:
    vector<string> fizzBuzz(int n) {
        vector<string> res;
        map<int, string> fizzBuzzDict = {
            {3, "Fizz"},
            {5, "Buzz"} 
        };
        
        for(int i=1;i<=n;i++){
            string tmp="";
            for(auto key:fizzBuzzDict){
                if(i%key.first==0){
                    tmp+=key.second;
                }
 
            }
            if(tmp==""){
                tmp+=to_string(i);
            }
            res.push_back(tmp);
        }
        return res;
    }
};
```

### 204. 计算质数
https://blog.csdn.net/yangxjsun/article/details/80201735
- 计算小于正整数n的质数数量
- 试除法：需要进行优化： 
    - 只对奇数进行判断
    - 对奇数x判断时：只尝试**除以从 3 到√x 的所有奇数**
    - 进一步优化到只要尝试**小于√x 的质数**即可
    - 时间复杂度：$O(n/2 log\sqrt{n})$,空间复杂度O(log\sqrt{n})
    - 根据素数范围公式：小于x的质数有$x/ln(x)$个

```class Solution {
vector<int> primes;
public:
    bool check(int n){
        if(primes.empty()){
            if(n>=2){
                if (n>2) primes.push_back(n);
                return true;
            }
            return false;
        }
        int i=0;
        //cout<<n<<endl;
        while((primes[i]*primes[i])<=n){
            //cout<<primes[i]<<endl;
            if(n%primes[i++]==0) return false;
        }
        primes.push_back(n);
        //cout<<n<<endl;
        return true;
       
    }
    int countPrimes(int n) {
        int res=0;
        if(n<=2) return 0;
        for(int i=2;i<n;i++){
            if(i==2) {
                res=1;
            }
            else{
                if(i%2){
                res=check(i)?res+1:res;
                }
            }
        }
        return res;
    }
};
```
- 筛选法: 厄拉多塞筛法，简称埃氏筛
从小到大遍历到数 xx 时，倘若它是合数，则它一定是某个小于 xx 的质数 yy 的整数倍，故根据此方法的步骤，我们在遍历到 yy 时，就一定会在此时将 xx 标记为 \textit{isPrime}[x]=0isPrime[x]=0。因此，这种方法也不会将合数标记为质数。

当然这里还可以继续优化，对于一个质数 xx，如果按上文说的我们从 2x2x 开始标记其实是冗余的，应该直接从 x\cdot xx⋅x 开始标记，因为 2x,3x,\ldots2x,3x,… 这些数一定在 xx 之前就被其他数的倍数标记过了，例如 22 的所有倍数，33 的所有倍数等

```class Solution {
public:
    int countPrimes(int n) {
        vector<int> isPrime(n, 1);
        int ans = 0;
        for (int i = 2; i < n; ++i) {
            if (isPrime[i]) {
                ans += 1;
                if ((long long)i * i < n) {
                    for (int j = i * i; j < n; j += i) {
                        isPrime[j] = 0;
                    }
                }
            }
        }
        return ans;
    }
};
 
```

### 326. 3的幂数

- 循环除法： 时间复杂度O(log3n) 空间复杂度O(1)
```class Solution {
public:
    bool isPowerOfThree(int n) {

        bool res=false;
        while(n%3==0 && n>0){
            n/=3;
        }
        if(n==1) res=true;
        return res; 
    }
};
```

- 截断法： 计算2^31 int型的最大范围内的幂数值： 3^19
- 空间复杂度O(1) 时间复杂度O(1)
```class Solution {
public:
    bool isPowerOfThree(int n) {

        bool res=false;
        // 3^19 = 1162261467 
        const int large=1162261467 ;
        return (n>0)&&(large%n==0);
    }
};
```

### 13. 罗马数字转整数

```class Solution {
map<char,int> dict;
public:
    int romanToInt(string s) {
        dict['I']=1;
        dict['V']=5;
        dict['X']=10;
        dict['L']=50;
        dict['C']=100;
        dict['D']=500;
        dict['M']=1000;
        int res=0;
        if (s.empty()) return res;
        for(int i=0;i<s.length()-1;i++){
            if(dict[s[i]]<dict[s[i+1]]){
                res-=dict[s[i]];
            }
            else{
                res+=dict[s[i]];
            }
        }
        res+=dict[s[s.length()-1]];
        return res;

    }
};

```


### 181. 位1的个数

- 通过位运算来统计1的个数，通过与&运算来统计每个位数是否为1
    - 时间复杂度分析： 位数有32的限制，因此时间复杂度在O(1)

```class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count=0;
        int mask=1;
        while(n!=0){
            if (mask&n) count++;
            n>>=1;
        }
        return count;    
    }
};
```

- 进一步优化： 官方思路(布赖恩·克尼根位计数算法)
    - 通过n与(n-1)的与运算
    - 能够减少循环次数
```class Solution {
public:
    int hammingWeight(uint32_t n) {
        int count=0;
        while(n!=0){
            count++;
            n&=(n-1);
        }   
        return count;
    }
};
```

### 444. 汉明距离计算
- 计算汉明距离：两个数字的二进制表示有多少位不同
- 通过XOR异或计算得到差异位，计算有多少个1即可
- 可以用上上面题中计算位1的算法
```class Solution {
public:
    int hammingDistance(int x, int y) {
        int target=x^y;
        //cout<<target<<endl;
        if(target<1)
            return 0;

        int count=0;
        while(target){
            if(target&1) count++;
            target>>=1;
            //count++;
        }
        
        return  count;
    }
};
```


#### Happy New Year!!!

### 180. 颠倒二进制位

- 基础思路：逐位处理
- 复杂度分析： O(log2N)  空间复杂度：O(1)
```class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t res=0;
        int count=31;
        while(n){
            if(n&1){
                res+=pow(2,count);
            } 
            n>>=1;
            count--;
        }
        return res;
        
    }
};
```

- 优化写法：把幂乘改成位操作<<

```class Solution {
  public:
  uint32_t reverseBits(uint32_t n) {
    uint32_t ret = 0, power = 31;
    while (n != 0) {
      ret += (n & 1) << power;
      n = n >> 1;
      power -= 1;
    }
    return ret;
  }
}
```

- 更常规的操作： 直接取模求和

- 分治合并：利用**位运算** 操作进行翻转（**掩码+位移+合并**）：
    - 先按照16位左右翻转
    - 再以8位左右翻转
    - ... 最后翻转到1位
    - 其中翻转使用有规模/模式的数字进行位操作
    - 当分成2个16位小块时，通过左右位移然后再通过或运算合并两部分结果
    - 当分成4个8位小块时，每部分先通过**掩码**来选取要翻转的位置，然后再通过8位位移操作来实现位置翻转，通过或运算合并结果
    - 掩码如：00001111000011111 0x0f0f0f0f这种格式实现指定位置的数据提取


```
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t res=n;
        res = (res<<16)|(res>>16);
        res = ((res& 0xff00ff00)>>8)|((res& 0x00ff00ff)<<8);
        res = ((res& 0xf0f0f0f0)>>4)|((res& 0x0f0f0f0f)<<4);
        res = ((res& 0xcccccccc)>>2)|((res& 0x33333333)<<2);
        res = ((res& 0xaaaaaaaa)>>1)|((res& 0x55555555)<<1);

        return res;
        
    }
};
```


### 118. 杨辉三角


- 时间复杂度分析：O（n(n+1)/2）

```
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> tri(numRows);
        if(numRows<1)
          return tri;
        tri[0].push_back(1);
        for(int i=1; i<numRows;i++){
            for(int j=0;j<i+1;j++){
                if(j==0| j==i)
                {
                    tri[i].push_back(tri[i-1][0]);
                }
                else{
                    tri[i].push_back(tri[i-1][j-1]+tri[i-1][j]);
                }
            }
        }
        return tri;
    }
};
``` 


### 2. 有效的括号

- 判断字符串内括号是否有效，强调**同级括号的封闭性**
- 使用**栈**结构对字符串处理
- 主要思路是：**利用栈压入左字符，当遇到右字符时进行pop**
    - **考虑临界情况**：当字符串长度为奇时，返回false；
    - 在pop时考虑栈内是否有字符压入，即右字符先出现的特殊情况；
- 复杂度分析： 时间复杂度：O(N)   空间复杂度 O(N+C) *有哈希表的影响*

```
class Solution {
public:
    bool isValid(string s) {
        stack<char> res;
        map<char, char> m1;
	    m1['}'] = '{';
	    m1[']'] = '[';
        m1[')'] = '(';
        for(auto chr:s){
            if(chr=='('|chr=='['|chr=='{'){
                res.push(chr);
            }
            else{
                if(res.size()==0){
                    return false;
                }
                char tmp=res.top();
                if(tmp==m1[chr]){
                    res.pop();
                }else{
                    return false;
                }
            }
        }
        if(res.size()) return false;
        return true;
    }
};
```

- 优化思路： 利用ANSCII码替代map处理，左右括号符合ANSCII码差异在1或者2。


### 缺失的数字
- 基础思路：先对数组排序，然后遍历数组，查找空缺位置；时间复杂度：排序算法复杂度(O(nlogN))
- 为了追求线性复杂度，利用更简单的算法：利用高斯求和进行计算：
    - 时间复杂度 O(N) 空间复杂度 O(1)
    - **没有考虑数据溢出的情况**，求和可能会造成数据溢出

```class Solution {
public:
    int missingNumber(vector<int>& nums) {
        if(nums.empty())
        return 0;
        int lens = nums.size();
        int count = (lens+1)*(lens)/2;
        int res=0;
        for(auto i:nums)
            res+=i;
        return count-res;

    }
};
```

- **优化改进**：可以通过边加边减的操作来改写代码，这种写法并不能避免极限情况下的溢出，当循环中的前两个数字为最大的两个时，可能直接就溢出了
    - 注意写法，避免数组越界
```
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        if(nums.empty())
        return 0;
        int res =  nums.size();
        for(int i=0;i<nums.size();i++){
            res+=i;
            res-=nums[i];
        }
        return res;

    }
};
```
- **最简单的优化，利用long型来存储**

- **位运算**操作，利用XOR异或运算来计算缺失值
    - 异或运算中 相同值异或为0，数组连续异或得到没有重复的数字。
    - `3^0^0^1^2^1^3=2` n*2+1个数字异或肯定得到一个不重复的数字，即缺失值
    - 时间复杂度O(N) 空间复杂度O(1)
```class Solution {
public:
    int missingNumber(vector<int>& nums) {
        if(nums.empty())
        return 0;
        int res =  nums.size();
        for(int i=0;i<nums.size();i++){
            res^=i;
            res^=nums[i];
        }
        return res;
    }
};
```

### 371. 利用位运算实现两整数之和 [Medium]
> 写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。


- 不能使用 + - 运算，因此可以想到使用位运算
- **位运算中的异或/与操作的特性**：
  - `^` 异或运算,可以**表示二进制下两个位置上的加和的无进位表示**
  - `&` 与运算，能够反映加和的进位情况，**即有无进位出现0/1**
- 因此总结上面的操作逻辑：
  - 第一步：相加各位的值，不算进位，得到010，二进制每位相加就相当于各位做异或操作，101^111。
  - 第二步：计算进位值，相当于各位进行与操作得到101，再向左移一位得到1010，(101&111)<<1。
  - 第三步重复上述两步 
  - **当没有进位产生时即停止操作**
  - 与十进制下的加法逻辑一致

```
class Solution {
public:
    int getSum(int a, int b) {
        while (b) {
            int c = a^b; // 异或操作 获取无进位加法的结果
            b = (unsigned int)(a&b)<<1;
            a = c;
        }
        return a;

    }
};
```

### 202. 快乐数
- *不是很快乐。。。。。*
- 解题关键： 如何解决无限循环
  - 在数的不断拆分中，可能出现循环的环结构，使用哈希表进行数的重复检测即可
- 时间复杂度 O(logN) 空间复杂度 O(logN)

```
class Solution {
public:
    int count(vector<int> arr) {
        int sum = 0;
        for (auto s : arr) {
            sum += s*s;
        }
        return sum;
    }
    
    bool isHappy(int n) {
        set<int> numbers;
        vector<int> w;
        numbers.insert(n);
        while (true) {
            while(n) {
                int t = n%10;
                w.push_back(t);
                n = n/10;
            }
            n = count(w);
            if (n == 1)
                return true;
            w.clear();
            if (numbers.count(n)) {
                return false;
            }
            numbers.insert(n);
        }
    }
};
```

### 172. 阶乘后的0的个数
- 数学题，如何避免阶乘的低效率？
- 把问题转换为计算因子5的数量
  - `5 10 15 20 25` 中分别有 `1,1,1,1，2`个5因子
  - 最后的结果可以计算除以`5 25 125...`的数量和

https://leetcode-cn.com/problems/factorial-trailing-zeroes/solution/liang-dao-lei-si-de-jie-cheng-ti-mu-xiang-jie-by-l/

```
class Solution {
public:
    int trailingZeroes(int n) {
        int ans = 0;
        int factor = 5;
        while (factor <= n) {
            ans += n / factor;
            factor *= 5;
        }
        return ans;
    }
};
```

### 793. 阶乘函数后K个零
- 给出目标值K，计算有多少个阶乘函数有K个尾零
- 利用上题中的解法可以得到n!的尾零个数
- 然后基于二分查找的思想，进行大数搜索，搜索出阶乘尾0等于K的左右边界
  - 因为这个阶乘函数的尾零个数是一个连续上升的函数，因此可以使用二分查找进行
  - 时间复杂度 O((logN)^2) 

```
class Solution {
public:
    long travZeros(long n) {
        long factor = 5;
        long ans = 0;
        while (factor <= n) {
            ans += n/factor;
            factor *= 5;
        }
        return ans;
    }        
    
    int binarySearch(int K, bool left_flag) {
        long left = 0;
        long right = LONG_MAX;
        while (left <= right) {
            long mid = left + (right - left)/2;
            cout << mid << " " << travZeros(mid) << endl;

            if (travZeros(mid) == K) {
                if (left_flag)
                    right = mid - 1;
                else 
                    left = mid + 1;
            }
            else if (travZeros(mid) < K) {
                left = mid + 1;
            }
            else if (travZeros(mid) > K) {
                right = mid - 1;
            }
        }
        return left_flag ? left : right;
    }

    int preimageSizeFZF(int K) {
        return binarySearch(K, false) - binarySearch(K, true) + 1;
    }
};
```


### 169. 多数元素 
- 返回数组中出现次数多于`n/2`的元素
- 基础做法：排序然后直接根据索引取元素即可
- 升级：使用哈希表，计算各个元素的出现频率，然后取最大频率对应的值即可
- 投票法：
  > 假设数组中每个不同的数字就代表一个国家，而数字的个数就代表这个国家的人数，他们在一起混战，就是每两个两个同归于尽。我们就可以知道那个人数大于数组长度一半的肯定会获胜。
  > 就算退一万步来说，其他的所有人都来攻击这个人数最多的国家，他们每两个两个同归于尽，最终剩下的也是那个众数

```
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int candidate = nums[0];
        int count = 1;
        for (int i = 1; i < nums.size(); i++) {
            if (count == 0) {
                count ++;
                candidate = nums[i];
            }
            else if (candidate == nums[i]) {
                count++;
            }
            else {
                count--;
            }
        }
        return candidate;
    }
};
```

### 621. 任务调度器 [Medium]
- 给出任务序列和同任务冷却时间，找出最小的任务序列完成时间
- 核心： 任务完成主要还是取决于次数最多的任务，**要根据次数最多的任务进行排期**
  - 根据冷却时间构成任务轮次，最后再加上跟次数最多的任务的任务数量即得到最终结果
  - 需要考虑临界情况，如**没有冷却时间**，对应的值可以调整一下最后取任务列表长度和目标值的最大值

```
class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        vector<int> bins(26);
        for (int i = 0; i < tasks.size(); i++) {
            bins[tasks[i] - 'A']++;
        }
        sort(bins.rbegin(), bins.rend());
        int maxBin = bins[0];
        int cur = (maxBin - 1)*(n + 1);
        int cnt = 0;
        for (int i = 0; i < bins.size(); i++) {
            if (bins[i] >= maxBin)
                cnt++;
            if (bins[i] == 0)
                break;
        }
        //cout << cnt << endl;
        int Len = tasks.size();
        return max(Len, cnt + cur);
    }
};
```

### 470. 用Rand7实现Rand10 *
- 如果是要用`rand10` 实现 `rand7` 直接取1～7之间的值即可，这种情况下取到的值也是等概率分布的
- 对于 `(randX()-1)*Y + randY()` 可以得到`[1, X*Y]`之间的等概率分布
- 本题也是利用这个基本理论，得到[1,49]之间的等概率分布，然后拒绝其中大于40的部分，剩下部分来得到rand10()
    - **可以对拒绝空间进行进一步优化，对于被拒绝的部分即rand9(), 可以得到  `[1,63]` **的概率分布空间
    - 继续对拒绝空间部分1~3进行利用 rand3(), 可以得到`[1,21]` 拒绝空间只有1，这样的采样效率会大幅提升
```
class Solution {
public:
    int rand10() {
        while(true) {
            // (randx() - 1)*Y + randY()
            int a = (rand7() - 1)*7 + rand7();
            if (a <= 40)
                return (a )%10 + 1;
            a = (a - 40 - 1)* 7 + rand7();
            if (a <= 60) 
                return (a )%10 + 1;
            a= (a - 60 - 1)* 7 + rand7();
            if (a <= 20) 
                return (a )% 10 + 1;
        }
        return 0;
    }
};
```