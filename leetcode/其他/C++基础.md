

[TOC]
### String类复现 [Tencent]

```
class String  
{  
public:  
    String(const char *str = NULL);// 普通构造函数    
    String(const String &other);// 拷贝构造函数    
    ~String(void);// 析构函数    
    String & operator = (const String &other);// 赋值函数    
private:  
    char *m_data;// 用于保存字符串    
};
```

- 实现构造函数及赋值函数
- 需要对C++的申请空间操作有一定基础
    - 构造函数：申请空间： `strcpy` 与 `memcpy` 前者用来copy字符串 ` *strcpy(char *dest,char *src);` **遇到'/0'就结束拷贝**
    - memcpy 用于做内存拷贝 `memcpy(b, a, sizeof(b))`
    - 需要先申请空间，然后再进行拷贝
```
//普通构造函数    
String::String(const char *str)  
{  
    if (str == NULL)  
    {  
        m_data = new char[1];// 得分点：对空字符串自动申请存放结束标志'\0'的，加分点：对m_data加NULL判断    
        *m_data = '\0';  
    }  
    else  
    {  
        int length = strlen(str);  
        m_data = new char[length + 1];// 若能加 NULL 判断则更好    
        strcpy(m_data, str);  
    }  
}  
```  
```  
// String的析构函数    
String::~String(void)  
{  
    delete[] m_data; // 或delete m_data;    
}  
```
- 拷贝构造函数： 
  - 即利用string对象来构造当前对象
  - `strlen()`来获取当前char的长度
  - 先申请空间，然后再同样进行`strcpy`
```
//拷贝构造函数    
String::String(const String &other)// 得分点：输入参数为const型    
{          
    int length = strlen(other.m_data);  
    m_data = new char[length + 1];//加分点：对m_data加NULL判断    
    strcpy(m_data, other.m_data);  
}  
```
- **赋值函数**
  - 应该算最复杂的
  - 首先要考虑是不是等于当前对象 `this == &other` **使用this指针来进行操作**
  - 然后需要考虑当前对象是否已有字符串，有的话需要进行空间释放`delete m_data`
  - 最后再进行空间申请和copy
  - **return *this** 需要返回string 对象
```
//赋值函数    
String & String::operator = (const String &other) // 得分点：输入参数为const型    
{  
    if (this == &other)//得分点：检查自赋值    
        return *this;   
    if (m_data)  
        delete[] m_data;//得分点：释放原有的内存资源    
    int length = strlen(other.m_data);  
    m_data = new char[length + 1];//加分点：对m_data加NULL判断    
    strcpy(m_data, other.m_data);  
    return *this;//得分点：返回本对象的引用      
} 
```