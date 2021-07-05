
- SQL面经： https://www.nowcoder.com/discuss/95812
- 
### 获得积分最多的人
> 查找积分增加最高的用户的名字(限定只有一个)，以及他的总积分是多少


- 关键api: 分组`group by`  排序`order by` `desc`
- 取目标行： `limit 1`
- 拼接： `join on` 
```SQL
select u.name, t.grade_sum 
from user u join (select user_id, sum(grade_num) grade_sum from 
      grade_info 
      group by user_id order by grade_sum desc 
      limit 1) t
on u.id = t.user_id
```
> SQL查找积分增加最高的用户的id(可能有多个)，名字，以及他的总积分是多少，查询结果按照id升序排序
- 使用`rank() over` 对总积分值进行排名，取排名为1的行
- `rank() over (order by grade_sum desc)`
- 降序：`desc` 升序: `asc`
```SQL
select u.id, u.name, ttt.grade_sum 
from user u join (select * from (
    select *, rank() over (order by grade_sum desc) as ranking from (
    select user_id, sum(grade_num) as grade_sum from 
      grade_info group by user_id order by grade_sum desc)t)tt 
                   where tt.ranking=1)ttt
on u.id = ttt.user_id
```
> 