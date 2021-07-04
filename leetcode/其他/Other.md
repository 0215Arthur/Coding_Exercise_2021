- [263. 丑数](#263-丑数)
- [264. 丑数II](#264-丑数ii)
- [313. 超级丑数](#313-超级丑数)
- [1201. 丑数III](#1201-丑数iii)
- [384. 数组打乱shuffle](#384-数组打乱shuffle)
- [412. Fizz Buzz](#412-fizz-buzz)
- [204. 计算质数](#204-计算质数)
- [326. 3的幂数](#326-3的幂数)
- [12. 整数转罗马数](#12-整数转罗马数)
- [13. 罗马数字转整数](#13-罗马数字转整数)
- [181. 位1的个数](#181-位1的个数)
- [444. 汉明距离计算](#444-汉明距离计算)
  - [Happy New Year!!!](#happy-new-year)
- [180. 颠倒二进制位](#180-颠倒二进制位)
- [371. 利用位运算实现两整数之和 [Medium]](#371-利用位运算实现两整数之和-medium)
- [202. 快乐数](#202-快乐数)
- [172. 阶乘后的0的个数](#172-阶乘后的0的个数)
- [793. 阶乘函数后K个零](#793-阶乘函数后k个零)
- [169. 多数元素](#169-多数元素)
- [621. 任务调度器 [Medium]](#621-任务调度器-medium)
- [470. 用Rand7实现Rand10 *](#470-用rand7实现rand10-)
- [1518. 换酒问题](#1518-换酒问题)
- [补充题：判断一个点是否在三角形内部 [美团/字节/百度***]](#补充题判断一个点是否在三角形内部-美团字节百度)
- [面试 16.03. 交点](#面试-1603-交点)
- [400. 第N个数](#400-第n个数)
- [528. 按权重随机选择](#528-按权重随机选择)


### 263. 丑数
> 给你一个整数 n ，请你判断 n 是否为 丑数 。如果是，返回 true ；否则，返回 false 。
> 丑数 **就是只包含质因数 2、3 和/或 5 的正整数**
**1 通常被视为丑数。
**
- 简单的数学判断题目
- 可以利用循环相除进行余数判断即可
  - 需要注意对特殊情况： 0的判断


```c++
class Solution {
public:
    bool isUgly(int n) {
        int factors[3] = {2, 3, 5};
        int i = 0;
        while (n > 0 && i < 3) {
            if (n % factors[i] == 0) {
                n /= factors[i];
            }
            else {
                i++;
            }
        }
        if (n == 1) return true;
        return false;
    }
};
```

### 264. 丑数II
> 给你一个整数 n ，请你找出并返回第 n 个 丑数 。
丑数 就是只包含质因数 2、3 和/或 5 的正整数。
- 承接[LC263.], 本题需要搜索第n个丑数，最简单的方法为暴力遍历，取第n个数
- 从算法角度考虑，进行搜索优化：
  - **使用优先队列进行处理**
  - 利用最小堆进行中间丑数维护，每次将2、3、5的倍数插入堆中，
  - 进行n次出堆操作，最后一次出堆即为第n个丑数
  - 堆维护了数字的有序性，每次出堆即得到升序的第i个丑数
  - 由于过程中可能出现重复值，使用哈希表进行去重控制
- 时间复杂度 O(NlogN) 
- 空间复杂度 O(N)
```c++
class Solution {
public:
    int nthUglyNumber(int n) {
        // 使用long 避免溢出
        priority_queue<long, vector<long>, greater<long>> q;
        int ugly = 1;
        vector<int> factors = {2, 3, 5};
        unordered_set<long> seen;
        q.push(ugly);
        for (int i = 0; i < n; i++) {
            long cur = q.top();
            ugly = cur;
            q.pop();
            for (auto p : factors) {
                if (!seen.count(p * cur)) {
                    q.push(p * cur);
                    seen.insert(p*cur);
                }
            }
        }
        return ugly;
    }
};
```
- 基于动态规划的思想进行搜索：
  - `dp[i]`定义： 第i个丑数
  - 状态转移： `dp[i] = min(dp[p1]*2, dp[p2]*3, dp[p3]*5)， 2 <= i <= n `
  - 其中`p1`  `p2` `p3` 为三种因子的数量指针： 代表的是第几个数的2倍、第几个数 3 倍、第几个数 5 倍
  - 本质上计算与最小堆的入堆/出堆逻辑相同：
    - **小顶堆的方法是先存再排，dp的方法则是先排再存**
  - 初始化： `dp[1] = 1`  `p1 = p2 = p3 = 1`
- 时间复杂度 O(N)  空间复杂度 o(N)
- 本题关键： **`如何维护有序性`**  **`三指针 + 动态规划`**
```c++
class Solution {
public:
    int nthUglyNumber(int n) {
        // 使用long 避免溢出
        vector<long> dp(n + 1);
        dp[1] = 1;
        int p1 = 1;
        int p2 = 1;
        int p3 = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = min(dp[p1]*2, min(dp[p2]*3, dp[p3]*5));
            // 各个因子指针进行同步比较更新
            if (dp[i] == dp[p1]*2) {
                p1++;
            }
            if (dp[i] == dp[p2]*3) {
                p2++;
            }
            if (dp[i] == dp[p3]*5) {
                p3++;
            }
        }
        return dp[n];
    }
};
```
### 313. 超级丑数
> 超级丑数是指其所有质因数都是长度为 k 的质数列表 primes 中的正整数
1 是任何给定 primes 的超级丑数。

```
输入: n = 12, primes = [2,7,13,19]
输出: 32 
解释: 给定长度为 4 的质数列表 primes = [2,7,13,19]，前 12 个超级丑数序列为：[1,2,4,7,8,13,14,16,19,26,28,32] 。
```

- 与计算第n个丑数的问题的区别在于，本题是一个质数列表，是对质数做了扩展，但是丑数的本质没有变：
  - **1 是任何给定 primes 的超级丑数**。
- 仍然采用最小堆或者多指针的动态规划进行实现

```C++
class Solution {
public:
    int nthSuperUglyNumber(int n, vector<int>& primes) {
        vector<int> ps(primes.size(), 1);
        vector<int> dp(n+1);
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[ps[0]] * primes[0];
            for (int j = 1; j < primes.size(); j++) {
                dp[i] = min(dp[ps[j]] * primes[j], dp[i]);
            }
            for (int j = 0; j < primes.size(); j++) {
                if (dp[i] == dp[ps[j]] * primes[j]) {
                    ps[j]++;
                }
            }
        }
        return dp[n];
    }
};
```


### 1201. 丑数III
> 给你四个整数：n 、a 、b 、c ，请你设计一个算法来找出第 n 个丑数。
丑数是可以被 a 或 b 或 c 整除的 正整数 。
```
输入：n = 3, a = 2, b = 3, c = 5
输出：4
解释：丑数序列为 2, 3, 4, 5, 6, 8, 9, 10... 其中第 3 个是 4。
```
- 注意： 此题中1 不是默认的丑数，此处为广义上的丑数
- 由于a b c因子是输入的， 那么该题目的范围应该是比较大的
- 区别与上面两题中标准的丑数定义，本题无法用上面的dp、堆来解题，一方面时间复杂度还是比较高，另一方面存在临界问题
- 本题的思路是采用**二分法+数学分析**来完成的
  - 查找第n个丑数，我们可以通过数学分析得到[0,X]范围内的丑数数量：
  - `X/a + x/b + X/c - X/lcm(a,b) - X/lcm(a, c) - X/lcm(b,c) + X/lcm(a,b,c)`
    - 即计算被因子整除的数量，其中要避免重复计算的情况，因此需要计算因子间的最小公倍数，根据容斥原理得到实际的丑数数量
    - 计算最小公倍数： `lcm(a,b)`
    - 计算最大公约数： `gcd(a,b)`
 
- 关键点： **`最小公倍数+丑数数量分析`**

```c++
class Solution {
public:
    int nthUglyNumber(int n, int a, int b, int c) {
        long left = 0;
        long right = 2e9;
        // 最小公倍数计算
        // 统一使用long  避免乘法计算溢出
        long la = a;
        long lb = b;
        long lc = c;
        long lab = lcm(la, lb);
        long lac = lcm(la, lc);
        long lbc = lcm(lb, lc);
        long labc = lcm(lab, lc);
        while ( left < right) {
            long mid = left + (right - left) / 2;
            long sum = mid / la + mid / lb + mid / lc - mid / lab - mid/lac -mid/lbc + mid/labc;
            if (sum < n) {
                left = mid + 1;
            }
            else {// sum >= n  
                right = mid;
            }
        }
        return left;
    }
};
```



### 384. 数组打乱shuffle
> 设计算法来打乱一个**没有重复元素**的数组。


- 两种思路： 暴力算法：每次从数组中随机选择一个数字处理，重复N次构成新数组，并移除原数组的相应元素(即保证无放回随机采样，采样空间为n!)
    - 时间复杂度：O(n^2) 
- **Fisher-Yates 洗牌算法**：每次迭代中，**生成一个范围在当前下标到数组末尾元素下标之间的随机整数** `rand() % (n - i) + i`
    - 时间复杂度: O(n) 使用rand()%()来生成索引，不需要额外的空间；
    - 理论上同样是n!的采样空间
- 关键点 ： **`洗牌算法`** 

```c++
class Solution {
public:
    vector<int> data;
    Solution(vector<int>& nums) {
        data = nums;
    }
    /** Resets the array to its original configuration and return it. */
    vector<int> reset() {
        return data;
    }
    
    /** Returns a random shuffling of the array. */
    vector<int> shuffle() {
        vector<int> nums(data);
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            int t = rand() % (n - i) + i ;
            swap(nums[i], nums[t]);
        }
        return nums;
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

### 12. 整数转罗马数

- 模拟转换过程即可
```c++
class Solution {
public:
    string toRoman(int num, char one, char five, char ten) {
        if (num <= 3) return string(num, one); //堆叠即可
        if (num == 4) return string("") + one + five; 
        if (num <= 8) return string("") + five + string(num - 5, one);
        if (num == 9) return string("") + one + ten;
        return "";
    }
    string intToRoman(int num) {
        return toRoman(num/1000, 'M', 0, 0) + toRoman((num%1000)/100, 'C', 'D', 'M') +
            toRoman((num%100)/10, 'X', 'L', 'C') + toRoman((num%10), 'I', 'V', 'X');
    }
};
```
### 13. 罗马数字转整数

```c++
class Solution {
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
> 「快乐数」定义为：
对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。如果 可以变为  1，那么这个数就是快乐数。

- 解题关键： 如何解决无限循环
  - 在数的不断拆分中，可能出现循环的环结构，使用哈希表进行数的重复检测即可
- 时间复杂度 O(logN) 空间复杂度 O(logN)

```c++
class Solution {
public:
    bool isHappy(int n) {
        unordered_set<int> mps;
        while (true) {
            int sum = 0;
            while (n) {
                int s = n % 10;
                n /= 10;
                sum += s*s;
            }
            if (sum == 1) 
                return true;
            if (mps.count(sum)) {
                return false;
            }
            mps.insert(sum);
            n = sum;
        }
        return false;
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

```c++
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
```c++
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
### 1518. 换酒问题
> 用 numExchange 个空酒瓶可以兑换一瓶新酒。你购入了 numBottles 瓶酒。如果喝掉了酒瓶中的酒，那么酒瓶就会变成空的。 **请你计算 最多 能喝到多少瓶酒。**

- 模拟计算即可
```c++
class Solution {
public:
    int numWaterBottles(int numBottles, int numExchange) {
        int res = numBottles;
        while (true) {
            int a = numBottles / numExchange;
            int b =  numBottles % numExchange;
            if (a == 0) break;
            res += numBottles / numExchange;
            numBottles = a + b;
        }
        return res;
    }
};
```

### 补充题：判断一个点是否在三角形内部 [美团/字节/百度***] 
> 在二维坐标系中，所有的值都是double类型，那么一个三角形可以由3个点来代表，给定3个点代表的三角形，再给定一个点(x, y)，判断(x, y)是否在三角形中
> https://www.nowcoder.com/questionTerminal/f9c4290baed0406cbbe2c23dd687732c

- 思路1： 根据三角形面积和公式进行计算
- 思路2: 根据向量叉乘方向来判断
  - 计算AO*AB的叉乘结果，将三条边看作是向量，计算三组向量叉乘，**判断叉乘方向是否都一致**
  - **如果都为同一方向，说明点在三角形内**
  - 叉乘公式： AO ❎ AB  = (x1 * y2 - x2 * y1) 
  - 叉乘结果反映了两个向量的位置关系，叉乘不可交换 

```c++
#include<iostream>
#include<vector>
using namespace std;

bool cross_product(pair<double, double> a, pair<double, double> b, pair<double, double> t) {
    
    double ans = (t.first - a.first) * (b.second - a.second)  - (t.second - a.second) * (b.first - a.first);
    return ans > 0;
}

int main() {
    vector<pair<double, double>> ans(4);
    for (int i = 0; i < 4; i++) {
        float x, y;
        cin >> x >> y;
        ans[i]={x,y};
    }
    
    if (cross_product(ans[0], ans[1], ans[3]) && cross_product(ans[1], ans[2], ans[3]) 
        && cross_product(ans[2], ans[0], ans[3]))
        cout << "Yes" << endl;
    else if (!cross_product(ans[0], ans[1], ans[3]) && !cross_product(ans[1], ans[2], ans[3]) 
        && !cross_product(ans[2], ans[0], ans[3]))
        cout << "Yes" << endl;
    else 
        cout << "No" << endl;
    
    return 0;
}
```

### 面试 16.03. 交点
> 给定两条线段（表示为起点start = {X1, Y1}和终点end = {X2, Y2}），如果它们有交点，请计算其交点，没有交点则返回空值。若有多个交点（线段重叠）则返回 X 值最小的点，X 坐标相同则返回 Y 值最小的点

- **繁琐的几何问题**
- 思路： 使用参数方程式来表示线段
  - $x = x_1 + t_1(x2 - x1)$ $y = y_1 + t_1(y2 - y1)$
- 先判断**对应斜率是否相等**
  - **相等情况下： 判断两个线段是否会有交点：**
  - 即将线段二的点带入参数方程，查看是否有解： 
  -  $x_3 = x_1 + t_1(x2 - x1);$ $y_3 = y_1 + t_1(y2 - y1)$
  -  $x_3 - x_1 / (x2 - x1) == (y_3 - y_1) / (y_2 - y_1)$ 通过乘法的形式来进行比较，以避免分母为0的情况
  - 会有交点的情况下，再去更新结果即可
- 若斜率不等，根据通式计算参数中的系数即可

```c++
class Solution {
public:
    vector<double> ans;
    void inside(int x1, int y1, int x2, int y2, int xk, int yk) {
        if ((x1 == x2|| (min(x1, x2) <= xk && xk <= max(x1, x2) )) &&
            (y1 == y2 || min(y1, y2) <= yk && yk <= max(y1, y2) ) ){
                if (ans.empty() || ans[0] > xk || (xk == ans[0] && yk < ans[1])) {
                    ans = {double(xk), double(yk)};
                }
        }
    }
    vector<double> intersection(vector<int>& start1, vector<int>& end1, vector<int>& start2, vector<int>& end2) {
        int x1 = start1[0], y1 = start1[1];
        int x2 = end1[0], y2 = end1[1];
        int x3 = start2[0], y3 = start2[1];
        int x4 = end2[0], y4 = end2[1];
        // 参数方程： x = tx1 + (1-t)(x2 - x1) y = ty1 + (1-t)(y2 -y1)
        //vector<double> ans;
        if ((y2 - y1) * (x4 - x3) == (y4 - y3) * (x2 - x1))  {
            // 线段斜率相等
            // 判断相等斜率的直线是否会有相交
             if ((y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1))  {
                  // x3 y3 在线段1内
                 inside(x1, y1, x2, y2, x3, y3);
                inside(x1, y1, x2, y2, x4, y4);
                inside(x3, y3, x4, y4, x1, y1);
                inside(x3, y3, x4, y4, x2, y2);
             }
        }
        else {
            // 线段斜率不想等的情况， 根据比值计算斜率
             
            double t1 = (double)(x3 * (y4 - y3) + y1 * (x4 - x3) - y3 * (x4 - x3) - x1 * (y4 - y3)) / ((x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1));

            double t2 = (double)(x1 * (y2 - y1) + y3 * (x2 - x1) - y1 * (x2 - x1) - x3 * (y2 - y1)) / ((x4 - x3) * (y2 - y1) - (x2 - x1) * (y4 - y3));
            // 判断 t1 和 t2 是否均在 [0, 1] 之间
            if (t1 >= 0.0 && t1 <= 1.0 && t2 >= 0.0 && t2 <= 1.0) {
                ans = {x1 + t1 * (x2 - x1), y1 + t1 * (y2 - y1)};
            }
        }
        return ans;
    }
};
```

### 400. 第N个数
> 在无限的整数序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...中找到第 n 位数字。

```
输入：11
输出：0
解释：第 11 位数字在序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... 里是 0 ，它是 10 的一部分。
```
- 题目乍一看有些费解，实际上是一个数学问题，计算这个连续正数序列中，第n位正数是什么。即考虑不同正数的位数和连续情况。
  - 具体数据位数分布情况如下：
  - 10～99 两位数的总位数： `10*1*9*2`: 通项：`10^(count)*9*(count+1)`
  - 通过上面的位数可以确定当前的目标位所在区间： `10^(count)`即有x个位的正数的起点：`10/100/1000...`
  - 确定区间后计算当前值具体在那个正数上即可
```
10 11 12 ... 98 99              --> 90   个数字每个数字占两位
100 101 102 ... 998 999         --> 900  个数字每个数字占三位
1000 1001 1002 ... 9998 9999    --> 9000 个数字每个数字占四位
```
- 关键点: **`分析总位数的通项`**
```c++
class Solution {
public:
    int findNthDigit(int n) {
        if (n < 10) {
            return n;
        }
        n -= 9;
        int count = 1;
        while (true) {
            
            long long num = pow(10, count) * 9 * (count + 1);
            //cout << num << " ";
            if (n > num) {
                n -= num;
                count++;
            }
            else {
                int t = n % (count + 1);
                int l = n / (count + 1);
                
                if (t == 0) {// 整除情况 取某值的最后一位
                    int cur = pow(10, count) + l - 1;
                    return cur % 10;
                }
                else {
                    
                    int cur = pow(10, count) + l;
                    //cout << t << " "<< l << " " <<cur;
                    return int(to_string(cur)[t - 1] - '0');
                }
                
            }
        }
        return 0;
    }
};
```
### 528. 按权重随机选择
> 给定一个正整数数组 w ，其中 w[i] 代表下标 i 的权重（下标从 0 开始），请写一个函数 pickIndex ，它可以随机地获取下标 i，选取下标 i 的概率与 w[i] 成正比

- 计算前缀和数组， 在**[0,sum-1]**生成随机数，然后在前缀和数组中搜索首个大于随机数的前缀和，即对应的下标
- 时间复杂度： **O(logN)**   
- 关键点： **前缀和 + 二分查找**
```c++
class Solution {
public:
    vector<int> prob;
    int probSum = 0;
    Solution(vector<int>& w) {
        int pred = 0;
        for (int i = 0; i < w.size(); i++) {
            prob.push_back(pred + w[i]);
            pred = prob.back();
        }
        probSum = prob.back();
    }
    
    int pickIndex() {
        int idx = rand() % probSum ; // 设置
        int left = 0;
        int right = prob.size() - 1;
        int ans = 0;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (prob[mid] <= idx) {
                left = mid + 1;
            }
            else {
                right = mid;
            }
        }
        // for (int i = 0; i < prob.size(); i++) {
        //     if (idx <= prob[i]) {
        //         return i;
        //     }
        // }
        return left;
    }
};
```