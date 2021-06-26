# Coding_Exercise_2021
 Program and Coding exercises for interview :blush: :sunny:

...
## ⛽️ 面试编程题目整理
注： 题目主要来源于Leetcode、 《剑指offer》 与《程序员面试金典》；  
少部分补充题来自于面经收集， 企业题库： https://codetop.cc/#/home

### 🥤 [排序算法](./leetcode/排序与搜索/sort.md#排序算法)
  - [基础排序算法梳理](./leetcode/排序与搜索/sort.md#比较排序-vs-非比较排序)
- [延伸题目](./leetcode/排序与搜索/sort.md#延伸题目)
  - [56. 合并区间 [medium]](./leetcode/排序与搜索/sort.md#56-合并区间-medium)
  - [剑指40. 最小的k个数 [*]](#剑指40-最小的k个数-)
  - [215. 数组中的第K个最大元素 [Medium]*](#215-数组中的第k个最大元素-medium)
  - [347. 前K个高频元素 [Medium]*](#347-前k个高频元素-medium)
      - [基于快速排序的思想](#基于快速排序的思想)
      - [基于堆的思想](#基于堆的思想)
  - [386. 整数的字典序 [Medium] [字节 *]](#386-整数的字典序-medium-字节-)
  - [补充题： 计算数组的小和 *](#补充题-计算数组的小和-)
  - [剑指51. 数组中的逆序对 *](#剑指51-数组中的逆序对-)
  - [179. 最大数](#179-最大数)
### 🍞 [链表](/leetcode/数组/初级链表.md#查找节点)
- [查找节点](/leetcode/数组/初级链表.md#查找节点)
  - [2. 两数相加](#2-两数相加)
  - [剑指22. 链表中倒数第k个节点](#剑指22-链表中倒数第k个节点)
  - [141. 环形链表操作](#141-环形链表操作)
  - [142. 环形链表2 [Medium]](#142-环形链表2-medium)
  - [160. 相交链表](#160-相交链表)
  - [876. 链表的中间节点](#876-链表的中间节点)
  - [234. 回文链表判断](#234-回文链表判断)
  - [328. 奇偶数组  [Medium]](#328-奇偶数组-medium)
- [链表删除节点](./leetcode/数组/初级链表.md#链表删除节点)
  - [19. 删除链表的倒数第N个节点 [Medium]](#19-删除链表的倒数第n个节点-medium)
  - [83. 删除排序链表中的重复元素](#83-删除排序链表中的重复元素)
  - [82. 删除排序链表中的重复元素II **](#82-删除排序链表中的重复元素ii)
  - [237. 删除链表指定节点](#237-删除链表指定节点)
- [链表翻转操作](./leetcode/数组/初级链表.md#链表翻转操作)
  - [206. 反转链表](#206-反转链表)
  - [局部反转： 反转链表前N个元素](#局部反转-反转链表前n个元素)
  - [局部反转：反转链表中第[m,n]个元素](#局部反转反转链表中第mn个元素)
  - [25. k个一组反转链表 [Hard*]](#25-k个一组反转链表-hard)
- [链表合并](./leetcode/数组/初级链表.md#链表合并)
  - [21. 合并两个有序链表](#21-合并两个有序链表)
  - [23. 合并K个升序链表 [Hard*]](#23-合并k个升序链表-hard)
- [混合操作](./leetcode/数组/初级链表.md#混合操作)
  - [86. 分隔链表](#86-分隔链表)
  - [143. 重排链表](#143-重排链表)
  - [146. LRU cache结构设计 [Medium]](#146-lru-cache结构设计-medium)
  - [补充题： 排序奇升偶降链表 [字节 *]](#补充题-排序奇升偶降链表-字节)
### ♨️ 数组与字符串

### 🥃 队列与栈
### 🍾️ 二叉树


### 🍶 [回溯算法](./leetcode/回溯算法/backtrack.md/#回溯算法)
- [基本思想](#基本思想)
- [套路框架](#套路框架)
- [17. 电话号码的字母组合 [Medium]](#17-电话号码的字母组合-medium)
- [22. 括号生成 [Medium]](#22-括号生成-medium)
- [39. 组合数 [Medium] [ByteDance]](#39-组合数-medium-bytedance)
- [46. 全排列 Permutations [MEDIUM]](#46-全排列-permutations-medium)
- [47. 全排列 II](#47-全排列-ii)
- [51. N皇后 I [HARD]](#51-n皇后-i-hard)
- [52. N皇后 2 [HARD]](#52-n皇后-2-hard)
- [x. N皇后思考](#x-n皇后思考)
- [78. 子集 [Medium]](#78-子集-medium)
- [79. 单词搜索 [Medium]](#79-单词搜索-medium)
- [93. 复原IP地址 [美团]](#93-复原ip地址-美团)
- [698. 划分为k个相等的子集 *](#698-划分为k个相等的子集-)

### 🍺 动态规划

### 🍷 [图结构](/图.md#拓扑排序)
- [拓扑排序](#拓扑排序)
- [207. 课程表 [Medium]](#207-课程表-medium)
- [210. 课程表 II](#210-课程表-ii)
## 🔋面试智力/概率题整理
### 🏀常见概率题 [字节](prob.md/#概率题)
- [1. 一根木棒，截成三截，组成三角形的概率](#1-一根木棒截成三截组成三角形的概率)
- [2. 抛硬币吃苹果的概率](#2-抛硬币吃苹果的概率)
- [3. 蚂蚁不相撞的概率](#3-蚂蚁不相撞的概率)
- [4. 扔筛子问题](#4-扔筛子问题)
- [5. 随机数生成](#5-随机数生成)
- [6. 随机数生成 II](#6-随机数生成-ii)
### 🏓️常见智力题 [字节/腾讯/]
  - [1. 扔鸡蛋 [扔铁球]](prob.md/#1-扔鸡蛋-扔铁球)
  - [2. 白鼠试毒药问题](#2-白鼠试毒药问题)
  - [3.](#3)
  - [4. 先手必胜策略问题：](#4-先手必胜策略问题)
  - [5. 蚂蚁爬树问题](#5-蚂蚁爬树问题)
  - [6. 瓶子换饮料问题](#6-瓶子换饮料问题)
  - [7. 在24小时里面时针分针秒针可以重合几次](#7-在24小时里面时针分针秒针可以重合几次)
  - [8. 找砝码问题](#8-找砝码问题)
  - [9. 找砝码问题2](#9-找砝码问题2)
  - [10. 生成随机数问题：](#10-生成随机数问题)
  - [11.赛马问题：](#11赛马问题)
  - [赛马II](#赛马ii)
  - [12. 烧香/绳子/其他时间问题](#12-烧香绳子其他时间问题)
  - [13. 掰巧克力问题 / 辩论问题](#13-掰巧克力问题--辩论问题)
  - [14. 一副扑克牌，平均分成三堆，大小王分在同一堆的概率是多大？](#14-一副扑克牌平均分成三堆大小王分在同一堆的概率是多大)
  - [15. 圆周率里面是否可以取出任意数字？](#15-圆周率里面是否可以取出任意数字)
  - [16. 倒水问题](#16-倒水问题)
  - [17. 老虎吃羊问题](#17-老虎吃羊问题)



```
Coding_Exercise_2021
├─ Bytedance
│  └─ 高频考题.md
├─ README.md
├─ io.md
├─ leetcode
│  ├─ .DS_Store
│  ├─ 其他
│  │  ├─ C++基础.md
│  │  ├─ Other.md
│  │  ├─ Other.txt
│  │  └─ 哈希表.md
│  ├─ 动态规划
│  │  ├─ 53-greedy.png
│  │  └─ 动态规划-初级.md
│  ├─ 回溯算法
│  │  └─ backtrack.md
│  ├─ 字符串
│  │  └─ 初级字符串.md
│  ├─ 排序与搜索
│  │  ├─ sort.md
│  │  └─ summary.png
│  ├─ 数组
│  │  ├─ 初级链表.md
│  │  ├─ 数组-中级.md
│  │  ├─ 数组-初级.md
│  │  └─ 补充.png
│  ├─ 查找
│  │  └─ 二分查找.md
│  ├─ 树结构
│  │  ├─ 102-02.png
│  │  ├─ 102.png
│  │  ├─ 108-01.png
│  │  ├─ dfs-0.png
│  │  ├─ dfs-1.png
│  │  ├─ inorder-0.png
│  │  ├─ sysm-0.png
│  │  ├─ 中级-树结构.md
│  │  └─ 初级树结构.md
│  └─ 队列与栈
│     ├─ quene_stack.md
│     └─ test.cpp
├─ prob.md
└─ 图.md

```