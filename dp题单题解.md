# T1
* [滑雪](https://www.luogu.com.cn/problem/P1434)

依次以每个点为起点进行记忆化搜索即可

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#include<math.h>
#include<vector>
#include<queue>
#define ll long long

inline ll read()
{
	ll x=0,f=1;
	char ch=getchar();
	while(!isdigit(ch))
	{
		if(ch=='-') f=-1;
		ch=getchar();
	}
	while(isdigit(ch))
	{
		x=(x<<1)+(x<<3)+ch-'0';
		ch=getchar();
	}
	return x*f;
}

const ll maxn=110;
ll n,m,ans;
ll dis[maxn][maxn],a[maxn][maxn],vis[maxn][maxn];
ll go[5][2]={{0,0},{0,1},{1,0},{-1,0},{0,-1}};

inline ll dfs(ll x,ll y)
{
	if(dis[x][y]) return dis[x][y];
	dis[x][y]=1;
	
	for(int i=1;i<=4;i++)
	{
		ll xx=x+go[i][0];
		ll yy=y+go[i][1];
		
		if(xx>0&&yy>0&&xx<=n&&yy<=m&&a[x][y]>a[xx][yy])
		{
			dfs(xx,yy);
			
			dis[x][y]=std::max(dis[xx][yy]+1,dis[x][y]);
		}
	}
	
	return dis[x][y];
}

int main(void)
{
	n=read(),m=read();
	
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<=m;j++)
		{
			a[i][j]=read();
		}
	}
	
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<=m;j++)
		{
			ans=std::max(dfs(i,j),ans);
		}
	}
	
	printf("%lld\n",ans);
	
	return 0;
}
```

# T2
* [食物链](https://www.luogu.com.cn/problem/P3183)

记忆化搜索

```cpp
#include<iostream>
#include<cstdio>
#include<algorithm>
#include<math.h>
#include<cstring>
#define ll long long
using namespace std;

inline ll read()
{
	ll x=0,f=1;
	char ch=getchar();
	while(!isdigit(ch))
	{
		if(ch=='-') f=-1;
		ch=getchar();
	}
	while(isdigit(ch))
	{
		x=(x<<1)+(x<<3)+ch-'0';
		ch=getchar();
	}
	return x*f;
 } 

const ll maxn=2e5+10;

ll tot,n,m;
ll head[maxn*2],f[maxn*2],out[maxn*2],in[maxn*2]; 

struct node
{
	ll u,v,nxt;
} s[maxn<<1];

inline void add(ll u,ll v)
{
	s[++tot].v=v;
	s[tot].nxt=head[u];
	head[u]=tot;
}

inline ll dfs(ll u)
{
	if(f[u]) return f[u];
	ll ans=0;
	
	if(!out[u]&&in[u]) ++ans;
	
	for(int i=head[u];i;i=s[i].nxt)
	{
		ll v=s[i].v;
		ans+=dfs(v);
	}
	
	return f[u]=ans;
}

int main(void)
{
	n=read(),m=read();
	
	for(int i=1;i<=m;i++)
	{
		ll u=read(),v=read();
		add(u,v);
		out[u]++;
		in[v]++;
	}
	
	ll ans=0;
	
	for(int i=1;i<=n;i++)
	{
		if(in[i]==0) ans+=dfs(i);
	}
	
	printf("%lld\n",ans);
	
	return 0;
} 
```

# T3
* [合唱队形](https://www.luogu.com.cn/problem/P1091)

处理以 $i$ 为结尾的最长上升子序列，再处理以 $i$ 为起点的最长下降子序列

枚举断点 $i$ 更新答案即可

```cpp
#include<iostream>
#include<cstdio>
#include<math.h>
#include<cstring>
using namespace std;

int a[200],b[200],c[200];

int main(void)
{
	int n,i,j,maxx;
	
	cin>>n;
	
	memset(b,0,sizeof(b));
	memset(c,0,sizeof(c));
	
	for(i=1;i<=n;i++)
	{
		cin>>a[i];
	}
	
	for(i=1;i<=n;i++)
	{
		b[i]=1;
		
		for(j=1;j<=i-1;j++)
		{
			if((a[i]>a[j])&&(b[j]+1>b[i]))
			{
				b[i]=b[j]+1;
			}
		}
	}
	
	for(i=n;i>=1;i--)
	{
		c[i]=1;
		
		for(j=i+1;j<=n;j++)
		{
			if((a[j]<a[i])&&(c[j]+1>c[i]))
			{
				c[i]=c[j]+1;
			}
		}
	}
	
	maxx=0;
	
	for(i=1;i<=n;i++)
	{
		if(b[i]+c[i]>maxx)
		{
			maxx=b[i]+c[i];
		}
	}
	
	cout<<n-maxx+1<<endl;
	
	return 0;
}
```

# T4
* [没有上司的舞会](https://www.luogu.com.cn/problem/P1352)

树形 dp

设 $f[i][0/1]$ 为上司不来 ($f[i][0]$) 或者来 ($f[i][1]$) 的最大快乐指数

若 $S$ 代表一个点 $i$ 的子节点集合

则有：
$$
f[i][0] += Max\{ f[j][0] , f[j][1] \} ( j \in S )
$$
$$
f[i][1] += f[j][0] (j \in S)
$$

进行树形 dp 即可

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<math.h>
#include<algorithm>
#include<vector>
#define ll long long
using namespace std;

const ll maxn=6e3+10;
ll n,p;
ll v[maxn],a[maxn];
ll f[maxn][2];
vector<ll> e[maxn];

inline void dfs(ll x)
{
	f[x][0]=0;
	f[x][1]=a[x];
	
	for(int i=0;i<e[x].size();i++)
	{
		ll y=e[x][i];
		dfs(y);
		f[x][0]+=max(f[y][0],f[y][1]);
		f[x][1]+=f[y][0];
	}
}

int main(void)
{
	scanf("%lld",&n);
	
	for(int i=1;i<=n;i++) scanf("%lld",a+i);
	
	for(int i=1;i<=n-1;i++)
	{
		ll k,l;
		scanf("%lld%lld",&l,&k);
		e[k].push_back(l);
		v[l]=1;
	}
	for(int i=1;i<=n;i++)
	{
		if(!v[i])
		{
			p=i;
			break;
		}
	}
	
	dfs(p);
	
	printf("%lld",max(f[p][0],f[p][1]));
}
```

# T5
* [传纸条](https://www.luogu.com.cn/problem/P1006)

显然，处理从 $(n,m)$ 到 $(1,1)$ 的路径与处理从 $(1,1)$ 到 $(n,m)$ 的路径的最小代价一样

设 $f[i][j][k][l]$ 为从 $(1,1)$ 到 $(m,n)$ ，两张纸条的经过的路径所需的最小代价

那么有转移为：
$$
f[i][j][k][l] = Max\{ f[i-1][j][k][l-1] , f[i-1][j][k-1][l] , f[i][j-1][k-1][l] , f[i][j-1][k][l-1]\} + a[i][j] + a[k][l]
$$

由于一个同学只会帮一次忙，那么两张纸条到 $(n,m)$ 的前一个格子就分别为 $(n,m-1)$ 和 $(n-1,m)$

最后输出 $f[n][m-1][n-1][m]$ 即可

然而可以通过前三位进行加减运算确定第四维，降至 $O(n^3)$

```cpp
#include<iostream>
#include<cstdio>
#include<math.h>
#include<cstring>
#include<algorithm>
#define ll long long
using namespace std;

const ll maxn=100;
ll m,n;
ll h[maxn][maxn];
ll f[maxn][maxn][maxn][maxn];

int main(void)
{
	scanf("%lld%lld",&m,&n);
	
	for(int i=1;i<=m;i++)
	{
		for(int j=1;j<=n;j++)
		{
			scanf("%lld",&h[i][j]);
		}
	}
	
	for(int i=1;i<=m;i++)
	{	
		for(int j=1;j<=n;j++)
		{
			for(int p=1;p<=m;p++)
			{
				for(int q=j+1;q<=n;q++)
				{
					f[i][j][p][q]=max(f[i][j][p][q],max(f[i-1][j][p][q-1],max(f[i][j-1][p-1][q],max(f[i][j-1][p][q-1],f[i-1][j][p-1][q]))))+h[i][j]+h[p][q];
				}
			}
		}
	}
	
	printf("%lld\n",f[m][n-1][m-1][n]);
	
	return 0;
}
```

# T6
* [【模板】最长公共子序列](https://www.luogu.com.cn/problem/P1439)

本题如果使用 $O(n^2)$ 的最长公共子序列求解算法显然不行

但是观察本题性质，两个序列都是同一个序列的不同排列，那么其所包含的元素就是一样的

那么我们可以对其中一个序列以另一个序列的元素顺序为标准进行离散化

之后只需要做一个最长上升子序列即可

使用贪心加二分优化可以做到 $O(n \log(n))$ 

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#include<math.h>
#include<vector>
#include<queue>
#include<map>
#define ll long long

const ll maxn=1e5+10;
ll p1[maxn],p2[maxn],sta[maxn];
ll n,top;
std::map<ll,ll> q;

int main(void)
{	
	scanf("%lld",&n);
	
	for(int i=1;i<=n;i++)
	{
		scanf("%lld",p1+i);
		q[p1[i]]=i;
	}
	for(int i=1;i<=n;i++)
	{
		scanf("%lld",p2+i);
		p2[i]=q[p2[i]];
	}
	
	for(int i=1;i<=n;i++)
	{
		if(p2[i]>sta[top]) sta[++top]=p2[i];
		else
		{
			ll l=1,r=top,mid;
			
			while(l<=r)
			{
				mid=l+r>>1;
				if(p2[i]>sta[mid]) l=mid+1;
				else r=mid-1;
			}
			sta[l]=p2[i];
		} 
	}
	
	printf("%lld\n",top);
	
	return 0;
} 
```

# T7
* [[PA2014]Bohater](https://www.luogu.com.cn/problem/P4025)

本题可以使用贪心进行求解

对于可以回血的怪物，我们要按照其打怪时扣血顺序从小到大进行计算

对于不能回血的怪物，我们要按照其打怪后回血顺序从大到小进行计算

如果按照这个方式不能打完所有的怪物，就输出 "NIE"

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#include<math.h>
#include<vector>
#include<queue>
#define ll long long

const ll maxn=1e5+10;
ll n,z,nh,nk;
struct node
{
	ll d,a,id;
} s[maxn],h[maxn],k[maxn];

inline bool cmpa(node a,node b)
{
	return a.a>b.a;
}

inline bool cmpd(node a,node b)
{
	return a.d<b.d;
}

int main(void)
{
	scanf("%lld %lld",&n,&z);
	
	for(int i=1;i<=n;i++)
	{
		scanf("%lld %lld",&s[i].d,&s[i].a);
		s[i].id=i;
		if(s[i].a-s[i].d>=0) h[++nh]=s[i];
		else k[++nk]=s[i];
	}
	
	std::sort(h+1,h+nh+1,cmpd);
	std::sort(k+1,k+nk+1,cmpa);
	
	for(int i=1;i<=nh;i++)
	{
		if(h[i].d>=z)
		{
			printf("NIE\n");
			return 0;
		}
		else z=z-h[i].d+h[i].a;
	}
	for(int i=1;i<=nk;i++)
	{
		if(k[i].d>=z)
		{
			printf("NIE\n");
			return 0;
		}
		else
		{
			z=z-k[i].d+k[i].a;
		}
	}
	
	printf("TAK\n");
	
	for(int i=1;i<=nh;i++) printf("%lld ",h[i].id);
	for(int i=1;i<=nk;i++) printf("%lld ",k[i].id);
	
	return 0;
}
```

# T8
* [能量项链](https://www.luogu.com.cn/problem/P1063)

区间 dp

对于环形问题，可以采用断环为链的解决方法

设 $f[i][j]$ 表示合并 (i,j) 之间的所有珠子所可以获得的最大能量

则：
$$
f[i][j] = Max\{f[i][k] + f[k+1][j] + a[i] \times a[k-1] \times a[j-1]\}
$$

$O(n^3)$ 处理即可

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<math.h>
#include<queue>
#define ll long long

const ll maxn=220;
ll n,ans;
ll a[maxn];
ll dp[maxn][maxn];

int main(void)
{
	scanf("%lld",&n);
	
	for(int i=1;i<=n;i++)
	{
		scanf("%lld",a+i);
	}
	for(int i=1;i<=n;i++)
	{
		a[i+n]=a[i];
	}
	
	for(int j=2;j<2*n;j++)
	{
		for(int i=j-1;j-i<n&&i;i--)
		{
			for(int k=i;k<j;k++)
			{
				dp[i][j]=std::max(dp[i][j],dp[i][k]+dp[k+1][j]+a[i]*a[k+1]*a[j+1]);
			}
			ans=std::max(ans,dp[i][j]);
		}
	}
	
	printf("%lld\n",ans);
	
	return 0;
}

```

# T9
* [选课](https://www.luogu.com.cn/problem/P2014)

有依赖的背包问题，树形 dp

如果课程 $j$ 是课程 $i$ 的先修课，那么从 $j$ 向 $i$ 连一条边

如果这个课程没有先修课，那么就从 $0$ 号点向它连一条边

我们 dfs 时就从 $0$ 号点出发即刻，由于 $0$ 号点也要考虑在内，所以选课数量 $m$ 要加 $1$ 

设 $S$ 为课程 $i$ 的先修课集合, $f[i][j]$ 表示考虑以 $i$ 为根节点的子树，选了 $j$ 门课的最大学分

则：
$$
f[i][j] = Max\{ f[i][j-k]+f[p][k]\} ( p \in S , k \leq j)
$$

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<math.h>
#include<vector>
#define ll long long
using namespace std;

const ll maxn=310;
ll n,m,tot;
ll s[maxn],head[maxn];
ll f[maxn][maxn];

struct node
{
	ll to,nxt;
} edge[maxn];

inline void add(ll from,ll to)
{
	edge[++tot].nxt=head[from];
	edge[tot].to=to;
	head[from]=tot;
}

inline void dfs(ll x)
{
	f[x][1]=s[x];
	for(int i=head[x];i;i=edge[i].nxt)
	{
		ll y=edge[i].to;
		
		dfs(y);
		
		for(int j=m;j>=1;j--)
		{
			for(int k=j-1;k>=1;k--)
			{
				f[x][j]=max(f[x][j],f[x][j-k]+f[y][k]);
			}
		}
	}
}

int main(void)
{
	scanf("%lld%lld",&n,&m);

	for(int i=1;i<=n;i++)
	{
		ll k;
		scanf("%lld%lld",&k,&s[i]);
		add(k,i);
	}
	
	m++;
	
	dfs(0);
	
	printf("%lld",f[0][m]);
	
	return 0;
}
```
# T10
* [石子合并](https://www.luogu.com.cn/problem/P1880)

与“能量项链”一题相似

采用断环为链的技巧

设 $f[i][j]$ 为合并 $(i,j)$ 之间的石子的最大得分

则：
$$
f[i][j] = Max\{f[i][k]+f[k+1][j]+sum(i,j)\}
$$

其中 $sum(i,j)$ 为 $i$ 到 $j$ 的石子数量的和

求最小得分同理

```cpp
#include<iostream>
#include<cstdio>
#include<algorithm>
#include<math.h>
#include<cstring>
#define ll long long
using namespace std;

ll n,m,maxx=-1,minx=99999999999;
ll a[2001],sum[2001];
ll f[2001][2001];

ll maxn()
{
	memset(f,0,sizeof(f));
	for(int i=1;i<=n;i++) f[i][i]=0;
	
	for(int p=1;p<n;p++)
	{
		for(int i=1,j=i+p;i<=2*n,j<=2*n;i++,j=i+p)
		{
			for(int k=i;k<=j-1;k++)
			{
				f[i][j]=max(f[i][j],f[i][k]+f[k+1][j]+sum[j]-sum[i-1]);
			}
		}
	}
	for(int i=1;i<=n;i++) maxx=max(maxx,f[i][n+i-1]);
	
	return maxx;
}

ll minn()
{
	memset(f,0,sizeof(f));
	
	for(int p=1;p<n;p++)
	{
		for(int i=1,j=i+p;i<2*n,j<2*n;i++,j=i+p)
		{
			f[i][j]=99999999;
			for(int k=i;k<=j-1;k++)
			{
				f[i][j]=min(f[i][j],f[i][k]+f[k+1][j]+sum[j]-sum[i-1]);
			}
		}
	}
	for(int i=1;i<=n;i++) minx=min(minx,f[i][n+i-1]);
	
	return minx;
}

int main(void)
{
	scanf("%lld",&n);
	
	for(int i=1;i<=n;i++) scanf("%lld",&a[i]);
	
	for(int i=1;i<=n+n;i++)
	{
		a[i+n]=a[i];
		sum[i]=sum[i-1]+a[i];
	}
	
	printf("%lld\n%lld\n",minn(),maxn());
	
	return 0;
}
```

# T11
* [[ZJOI2006]三色二叉树](https://www.luogu.com.cn/problem/P2585)

首先，阅读题目，我们发现所给出的二叉树序列为对应的节点为其原树的先序遍历所得结果，而且对只有一子节点的点，该子节点为左子节点或者是右子节点无所谓

所以我们可以通过 dfs 先将其原树的形态处理出来，默认其只有一个子节点的点的子节点为左子节点

设 $f[x][0/1/2]$ 表示当前以 $i$ 为根的子树中， $i$ 的颜色为 $0/1/2$ 时的最大数量

其中 $0$ 表示为绿色点， $1/2$ 分别表示红色和蓝色

而题目中对于染色情况的限制，就意味着一个点和它的子节点以及这些子节点之间的颜色都不一样

那么有：
$$
f[i][0] = Max\{f[ls(i)][1]+f[rs(i)][2],f[ls(i)][2]+f[rs(i)][1]\} + 1
$$
$$
f[i][1] = Max\{f[ls(i)][0]+f[rs(i)][2],f[ls(i)][2]+f[rs(i)][0]\}
$$
$$
f[i][2] = Max\{f[ls(i)][1]+f[rs(i)][0],f[ls(i)][0]+f[rs(i)][1]\}
$$

最小值同理

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#include<math.h>
#include<vector>
#include<queue>
#define ll long long
#define ls(x) son[x][0]
#define rs(x) son[x][1]

const ll maxn=5e5+10;
char s[maxn];
ll n,tot;
ll son[maxn][2],f[maxn][3],g[maxn][3];

inline ll dfs()
{
	ll bh=++tot;
	if(s[bh]=='1') son[bh][0]=dfs();
	if(s[bh]=='2') son[bh][0]=dfs(),son[bh][1]=dfs();
	return bh;
}

inline void dp(ll x)
{
	if(ls(x)) dp(ls(x));
	if(rs(x)) dp(rs(x));
	if(!ls(x)&&!rs(x))
	{
		f[x][0]=g[x][0]=1;
		f[x][1]=g[x][1]=f[x][2]=g[x][2]=0;
	}
	
	f[x][0]=std::max(f[ls(x)][1]+f[rs(x)][2],f[ls(x)][2]+f[rs(x)][1])+1;
	f[x][1]=std::max(f[ls(x)][0]+f[rs(x)][2],f[ls(x)][2]+f[rs(x)][0]);
	f[x][2]=std::max(f[ls(x)][0]+f[rs(x)][1],f[ls(x)][1]+f[rs(x)][0]);
	g[x][0]=std::min(g[ls(x)][1]+g[rs(x)][2],g[ls(x)][2]+g[rs(x)][1])+1;
	g[x][1]=std::min(g[ls(x)][0]+g[rs(x)][2],g[ls(x)][2]+g[rs(x)][0]);
	g[x][2]=std::min(g[ls(x)][0]+g[rs(x)][1],g[ls(x)][1]+g[rs(x)][0]);
}

int main(void)
{
	scanf("%s",s+1);
	n=strlen(s+1);
	
	dfs();
	dp(1);
	
	printf("%lld %lld\n",std::max(std::max(f[1][0],f[1][1]),f[1][2]),std::min(std::min(g[1][0],g[1][1]),g[1][2]));
	
	return 0;
}
```
# T12
* [CF245H Queries for Number of Palindromes](https://www.luogu.com.cn/problem/CF245H)

处理回文串，我们可以枚举一个回文串的中间点，进行预处理

只需要将字符串分为长度为奇数的回文串和长度为偶数的回文串即可

之后，将区间 $(i,j)$ 看作一个点的坐标 $(i,j)$

即可处理一个二维前缀和，每次直接查询即可

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#include<math.h>
#include<vector>
#include<queue>
#define ll long long

const ll maxn=5e3+10;
char s[maxn];
ll dp[maxn][maxn];
ll n,q,l,r;

int main(void)
{
	scanf("%s",s+1);
	n=strlen(s+1);
	
	for(int i=1;i<=n;i++)
	{
		for(int j=i,k=i;j&&k<=n&&s[j]==s[k];j--,k++) dp[j][k]++;
		for(int j=i,k=i+1;j&&k<=n&&s[j]==s[k];j--,k++) dp[j][k]++;
	}
	
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<=n;j++)
		{
			dp[i][j]=dp[i][j]+dp[i][j-1]+dp[i-1][j]-dp[i-1][j-1];
		}
	}
	
	scanf("%lld",&q);
	
	for(int i=1;i<=q;i++)
	{
		scanf("%lld %lld",&l,&r);
		printf("%lld\n",dp[r][r]+dp[l-1][l-1]-dp[l-1][r]-dp[r][l-1]);
	}
	
	return 0;
}
```

# T13
* [[SCOI2005]互不侵犯](https://www.luogu.com.cn/problem/P1896)

状压 dp 经典例题

首先，可以预处理出对于同一行内可行的摆放情况，并记录该情况中摆放了多少个国王

对第一行的摆放情况进行预处理，之后每一行进行转移

设 $f[i][sum][S]$ 为对于第 $i$ 行，摆放的国王数量为 $sum$ 个时的摆放情况为 $S$

虽然 $S$ 可能会很大，但是之前对同一行内的摆放情况的预处理中可以发现这些合法的情况的数量其实并不多，那么我们可以对这些合法情况进行记录，每次直接查询即可

设同一行之间的摆放情况记录在数组 $vis$ 中，转移为：
$$
f[i][sum+p][j] = f[i][sum+p][j] + f[i-1][sum][k]
$$
$$
p+sum \leq K
$$

其中 $j$  ， $k$ 满足 $vis[j]$ 和 $vis[k]$ 在二进制表示中在相同位置或者相邻位置上不能同时为 $1$ 

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#include<math.h>
#include<vector>
#include<queue>
#define ll long long

const ll maxn=12;
ll n,m,cnt,sum,ans;
ll vis[(1<<maxn)],js[(1<<maxn)];
ll dp[maxn][maxn*maxn][(1<<maxn)];

int main(void)
{
	scanf("%lld %lld",&n,&m);
	
	for(int i=0;i<(1<<n);i++)
	{
		if((!((i<<1)&i))&&(!((i>>1)&i)))
		{
			vis[++cnt]=i;
			
			ll x=i;
			
			while(x)
			{
				js[cnt]+=x%2;
				x>>=1;
			}
		}
	}
	
	for(int i=1;i<=cnt;i++)
	{
		if(js[i]<=m) dp[1][js[i]][i]=1;
	}
	
	for(int i=2;i<=n;i++)
	{
		for(int j=1;j<=cnt;j++)
		{
			for(int k=1;k<=cnt;k++)
			{
				if(vis[j]&vis[k]) continue;
				if(vis[j]&(vis[k]<<1)) continue;
				if((vis[j]<<1)&vis[k]) continue;
				
				for(int p=1;p<=m;p++)
				{
					if(js[j]+p<=m)
					{
						dp[i][js[j]+p][j]+=dp[i-1][p][k];
					}
				}
			}
		}
	}
	
	for(int i=1;i<=n;i++)
	{
		for(int j=1;j<=cnt;j++)
		{
			ans+=dp[i][m][j];
		}
	}
	
	printf("%lld\n",ans);
	
	return 0;
}
```

# T14
* [[USACO11OPEN]Mowing the Lawn G](https://www.luogu.com.cn/problem/P2627)

单调队列优化 dp

初始设状态为： $f[i]$ 表示前 $i$ 只奶牛能够得到的最大的效率

因为从 $i-k$ 到 $i$ 中必然有一只奶牛是不选的，那么我们可以枚举这个断点进行处理

那么转移为：
$$
dp[i] = Max \{ dp[j-1]+ sum(j+1,i) \} 
$$
$$
i-k \leq j \leq i
$$

其中 $sum(j+1,i)$ 为从 $j+1$ 只牛到第 $i$ 只牛的效率和

那么对于 $sum$ 的计算我们可以维护一个前缀和处理

我们将其写开并简单处理： $dp[i] = Max \{dp[j-1] - sum[j] \} + sum[i]$

那么对于这个式子，可以用单调队列维护 $dp[j-1]-sum[j]$ 的值进行优化

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#include<math.h>
#include<vector>
#include<queue>
#define ll long long

const ll maxn=1e5+10;
ll n,k,head=0,tail=1;
ll e[maxn],s[maxn],dp[maxn];
ll q[maxn],p[maxn];

inline ll maxx(ll x)
{
	q[x]=dp[x-1]-s[x];
	
	while(head<=tail&&q[p[tail]]<q[x]) tail--;
	
	p[++tail]=x;
	
	while(head<=tail&&p[head]<x-k) head++;
	
	return q[p[head]];
}

int main(void)
{
	scanf("%lld%lld",&n,&k);
	for(int i=1;i<=n;i++)
	{
		scanf("%lld",e+i);
		s[i]=s[i-1]+e[i];
	}
	for(int i=1;i<=n;i++) dp[i]=maxx(i)+s[i];
	
	printf("%lld\n",dp[n]);
	
	return 0;
}
```

ps: 这个题可以 O2 暴力 $n^2$ 过百万

# T15
* [选择数字](https://www.luogu.com.cn/problem/P2034)

此题同 T14

# T16
* [[HNOI2010]合唱队](https://www.luogu.com.cn/problem/P3205)

区间 dp ， 记得取模

设 $f[i][j][0]$ 表示从第 $i$ 个人到第 $j$ 个人，其中第 $i$ 个人从左边进入的最大方案数

设 $f[i][j][1]$ 表示从第 $i$ 个人到第 $j$ 个人，其中第 $j$ 个人从右边进入的最大方案数

那么如果 $a[i]<a[i+1]$ ，按照题目要求，则 $f[i][j][0] + = f[i+1][j][0]$ ，同理，我们可以推导出全部的转移情况

对于初始化，我们只需要对一侧的入队情况进行处理即可

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#include<math.h>
#include<vector>
#include<queue>
#define ll long long

const ll maxn=1e3+10;
const ll mod=19650827;
ll n;
ll a[maxn],f[maxn][maxn][2];

int main(void)
{
	scanf("%lld",&n);
	for(int i=1;i<=n;i++) scanf("%lld",a+i);
	for(int i=1;i<=n;i++) f[i][i][1]=1;
	for(int len=1;len<=n;len++)
	{
		for(int i=1;i+len<=n;i++)
		{
			int j=i+len;
			
//			printf("%d %d %d %lld %lld\n",len,i,j,f[i][j][0],f[i][j][1]);
			
			if(a[i]<a[i+1]) (f[i][j][0]+=f[i+1][j][0])%=mod;
			if(a[i]<a[j]) (f[i][j][0]+=f[i+1][j][1])%=mod;
			if(a[j]>a[i]) (f[i][j][1]+=f[i][j-1][0])%=mod;
			if(a[j]>a[j-1]) (f[i][j][1]+=f[i][j-1][1])%=mod;
		}
	}
	
	printf("%lld\n",(f[1][n][0]+f[1][n][1])%mod);
	
	return 0;
}
```

# T17
* [[SCOI2009] windy 数](https://www.luogu.com.cn/problem/P2657)

数位 dp 裸题

可以直接记忆化搜索处理

或者是先进行预处理，然后求解即可，因为求 $[a,b]$ 并不容易计算，所以可以进行前缀和处理，之后相减求解

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#include<math.h>
#include<vector>
#include<queue>
#define ll long long

ll a,b;
ll s[20],f[20][20];

inline ll aabs(ll x)
{
	return x<0 ? -x : x;
}

inline void dfs()
{
	for(int i=0;i<=9;i++) f[1][i]=1;
	
	for(int i=2;i<=10;i++)
	{
		for(int j=0;j<=9;j++)
		{
			for(int k=0;k<=9;k++)
			{
				if(aabs(j-k)>=2) f[i][j]+=f[i-1][k];
			}
		}
	}
}

inline ll calc(ll x)
{
	memset(s,0,sizeof(s));
	ll len=0,ans=0;
	
	while(x)
	{
		s[++len]=x%10;
		x/=10;
	}
	
	for(int i=1;i<len;i++)
	{
		for(int j=1;j<=9;j++)
		{
			ans+=f[i][j];
		}
	}
	
	for(int i=1;i<s[len];i++)
	{
		ans+=f[len][i];
	}
	
	for(int i=len-1;i>=1;i--)
	{
		for(int j=0;j<=s[i]-1;j++)
		{
			if(aabs(j-s[i+1])>=2) ans+=f[i][j];
		}
		if(aabs(s[i+1]-s[i])<2) break;
	}
	
	return ans;
} 

int main(void)
{
	scanf("%lld %lld",&a,&b);
	
	dfs();
	
	printf("%lld\n",calc(b+1)-calc(a));
	
	return 0;
}
```
# T18
* [花神的数论题](https://www.luogu.com.cn/problem/P4317)

数位 dp ++

枚举数中包含 $1$ 的个数的情况，计算符合条件的数的数量，用快速幂计算

记忆化搜索即可处理

```cpp
#include<iostream>
#include<cstdio>
#include<algorithm>
#include<cmath>
#include<queue>
#include<cstring>
#define ll long long
using namespace std;
const ll mod=10000007;
ll n,len,ans=1;
ll top[100];
ll f[55][100][100][2];
ll dfs(ll now,ll tar,ll sum,bool lim)
{
	if(!now) return sum==tar;
	if(~f[now][tar][sum][lim]) return f[now][tar][sum][lim];
	ll up=lim ? top[now] : 1ll;
	ll anss=0;
//	printf("up=%d\n",up);
	for(ll j=0;j<=up;j++){//sum?? 
		anss = anss + dfs(now-1, tar , sum+(j==1), lim && (j==up));
	}
	return f[now][tar][sum][lim]=anss;
}
ll ksm(ll x,ll p){
	ll summ=1;
	while(p){
		if(p&1) summ=summ*x%mod;
		x=x*x%mod;
		p>>=1;
	}
	return summ;
}
int main()
{
	scanf("%lld",&n);
	while(n)
	{
		top[++len]=n&1;
		n>>=1;
	}
	for(ll k=1;k<=50;k++){
		memset(f,-1,sizeof(f));
		ans=(ans*(ksm(k,dfs(len,k,0,1))%mod))%mod;
	}
	printf("%lld\n",ans);
	return 0;
}

```

