
### C++ input out 【适应牛客网编程】




```
#include<iostream>
using namespace std;
int main() {
    int N;
    cin >> N;
    while (N--) {
        int a, b;
        cin >> a >>b;
        cout<< a + b << endl;
    }
    return 0;
}
```


```
#include <iostream>
using namespace std;

int main() {
    int t;
    while (true) {
        cin >> t;
        if (!t) 
           return 0;
        int sum = 0;
        while(t--) {
            int a;
            cin >> a;
            sum += a;
        }
        cout << sum << endl;
    }
    return 0;
}
```

> 输入数据有多组, 每行表示一组输入数据。每行不定有n个整数，空格隔开。(1 <= n <= 100)。

```
#include<iostream>
using namespace std;
int main()
{
    int a, sum;
    while( cin>>a ) {
        sum = 0;
        sum += a;
        while( cin.get() != '\n' ){
            cin >> a;
            sum += a;
        }
        cout<<sum<<endl;
    }
    return 0;
}
```

#### 字符串的情况
- 掌握cin.get() == ‘\n‘
- 掌握用vector<string>处理输入的string

```
#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
using namespace std;
int main() {
    int N;
    cin >> N;
    vector<string> res;
    while (N--) {
        string tmp;
        cin >> tmp;
        res.push_back(tmp);
    }
    sort(res.begin(),res.end());
    for (int i = 0; i < res.size(); i++) {
        if (i == res.size() - 1) {
            cout << res[i] << endl;
        }
        else 
            cout << res[i] << ' ';
    }
    return 0;
}

```

```
#include<iostream>
#include<vector>
#include<string>
#include<algorithm> 
using namespace std;

int main() {
    string s;
    vector<string> ans;
    while ( cin >> s) {
        ans.push_back(s);
        if (cin.get()== '\n') {
            sort(ans.begin(), ans.end());
            for(int i=0;i<ans.size();i++)
            {
                cout<<ans[i]<<' ';
            }
            cout<< endl;
            ans.clear();
        }
    }
    return 0;
}
```