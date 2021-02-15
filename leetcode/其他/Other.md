
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