# [论文笔记：多标签学习综述（A review on multi-label learning algorithms）](https://www.cnblogs.com/liaohuiqiang/p/9339996.html)

**2014 TKDE(IEEE Transactions on Knowledge and Data Engineering)**
**张敏灵，周志华**

**简单介绍**
传统监督学习主要是单标签学习，而现实生活中目标样本往往比较复杂，具有多个语义，含有多个标签。本综述主要介绍了多标签学习的一些相关内容，包括相关定义，评价指标，8个多标签学习算法，相关的其它任务。

**论文大纲**

1. 相关定义：学习任务，三种策略
2. 评价指标：基于样本的评价指标，基于标签的评价指标
3. 学习算法：介绍了8个有代表性的算法，4个基于问题转化的算法和4个基于算法改进的算法
4. 相关任务：多实例学习，有序分类，多任务学习，数据流学习

**相关定义**

1. 学习任务
   𝑋=ℝ𝑑X=Rd表示d维的输入空间，𝑌={𝑦1,𝑦2,...,𝑦𝑞}Y={y1,y2,...,yq}表示带有q个可能标签的标签空间。
   训练集$D = {(x^i, y^i)| 1 \leq i \leq m} ，𝑚表示训练集的大小，上标表示样本序数，有时候会省略。，m表示训练集的大小，上标表示样本序数，有时候会省略。x^i \in X，是一个𝑑维的向量。，是一个d维的向量。y^i \subseteq Y，是，是Y的一个标签子集。任务就是要学习一个多标签分类器的一个标签子集。任务就是要学习一个多标签分类器h(\cdot )，预测，预测h(x) \subseteq Y作为𝑥的正确标签集。常见的做法是学习一个衡量𝑥和𝑦相关性的函数作为x的正确标签集。常见的做法是学习一个衡量x和y相关性的函数f(x, y_j)，希望，希望f(x, y_{j1}) > f(x, y_{j2})，其中，其中y_{j1} \in y, y_{j2} \notin y。。h(x)可以由可以由f(x)衍生得到，衍生得到，h(x) = {y_j | f(x,y_j) > t(x), y_j \in Y}。。t(x)扮演阈值函数的角色，把标签空间对分成相关的标签集和不相关的标签集。阈值函数可以由训练集产生，可以设为常数。当扮演阈值函数的角色，把标签空间对分成相关的标签集和不相关的标签集。阈值函数可以由训练集产生，可以设为常数。当f(x, y_j)$返回的是一个概率值时，阈值函数可设为常数0.5。
2. 三种策略
   多标签学习的主要难点在于输出空间的爆炸增长，比如20个标签，输出空间就有220220，为了应对指数复杂度的标签空间，需要挖掘标签之间的相关性。比方说，一个图像被标注的标签有热带雨林和足球，那么它具有巴西标签的可能性就很高。一个文档被标注为娱乐标签，它就不太可能和政治相关。有效的挖掘标签之间的相关性，是多标签学习成功的关键。根据对相关性挖掘的强弱，可以把多标签算法分为三类。

- 一阶策略：忽略和其它标签的相关性，比如把多标签分解成多个独立的二分类问题（简单高效）。
- 二阶策略：考虑标签之间的成对关联，比如为相关标签和不相关标签排序。
- 高阶策略：考虑多个标签之间的关联，比如对每个标签考虑所有其它标签的影响（效果最优）。

**评价指标**
可分为两类

- 基于样本的评价指标（先对单个样本评估表现，然后对多个样本取平均）
- 基于标签的评价指标（先考虑单个标签在所有样本上的表现，然后对多个标签取平均）

每类又可分为用于分类任务和用于排序任务的指标，具体指标如下图所示
![img](https://images2018.cnblogs.com/blog/1160281/201807/1160281-20180720101415792-1227086943.png)

下面对图中的每个指标进行介绍。

**基于样本的评价指标**

1. Subset Accuracy（衡量正确率，预测的样本集和真实的样本集完全一样才算正确。）





1𝑝∑𝑖=1𝑝1{ℎ(𝑥𝑖)=𝑦𝑖}1p∑i=1p1{h(xi)=yi}



其中p表示测试集的样本大小，1{𝜋}1{π}表示𝜋π为真时返回1，否则返回0。

1. Hamming Loss（衡量的是错分的标签比例，正确标签没有被预测以及错误标签被预测的标签占比）





1𝑝∑𝑖=1𝑝1𝑞∣∣ℎ(𝑥𝑖)Δ𝑦𝑖∣∣1p∑i=1p1q|h(xi)Δyi|



其中ΔΔ表示两个集合的对称差，返回只在其中一个集合出现的那些值。

1. Accuracy, Precision, Recall, F值（单标签学习中准确率，精准率，召回率，F值的天然拓展）





𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦(ℎ)=1𝑝∑𝑖=1𝑝∣∣ℎ(𝑥𝑖)∩𝑦𝑖∣∣∣∣ℎ(𝑥𝑖)∪𝑦𝑖∣∣Accuracy(h)=1p∑i=1p|h(xi)∩yi||h(xi)∪yi|







𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛(ℎ)=1𝑝∑𝑖=1𝑝∣∣ℎ(𝑥𝑖)∩𝑦𝑖∣∣∣∣ℎ(𝑥𝑖)∣∣Precision(h)=1p∑i=1p|h(xi)∩yi||h(xi)|







𝑅𝑒𝑐𝑎𝑙𝑙(ℎ)=1𝑝∑𝑖=1𝑝∣∣ℎ(𝑥𝑖)∩𝑦𝑖∣∣∣∣𝑦𝑖∣∣Recall(h)=1p∑i=1p|h(xi)∩yi||yi|







𝐹𝛽(ℎ)=(1+𝛽2)⋅𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛(ℎ)⋅𝑅𝑒𝑐𝑎𝑙𝑙(ℎ)𝛽2⋅𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛(ℎ)⋅𝑅𝑒𝑐𝑎𝑙𝑙(ℎ)Fβ(h)=(1+β2)⋅Precision(h)⋅Recall(h)β2⋅Precision(h)⋅Recall(h)



1. One-error（度量的是：“预测到的最相关的标签” 不在 “真实标签”中的样本占比。值越小，表现越好）





𝑜𝑛𝑒−𝑒𝑟𝑟𝑜𝑟(𝑓)=1𝑝∑𝑖=1𝑝1{[𝑎𝑟𝑔𝑚𝑎𝑥𝑦𝑗∈𝑌𝑓(𝑥𝑖,𝑦𝑗)]∉𝑦𝑖}one−error(f)=1p∑i=1p1{[argmaxyj∈Yf(xi,yj)]∉yi}



1. Coverage（度量的是：“排序好的标签列表”平均需要移动多少步，才能覆盖真实的相关标签集）





𝑐𝑜𝑣𝑒𝑟𝑎𝑔𝑒(𝑓)=1𝑝∑𝑖=1𝑝𝑚𝑎𝑥𝑦𝑗∈𝑦𝑖𝑟𝑎𝑛𝑘𝑓(𝑥𝑖,𝑦𝑗)−1coverage(f)=1p∑i=1pmaxyj∈yirankf(xi,yj)−1



其中𝑟𝑎𝑛𝑘𝑓(𝑥𝑖,𝑦𝑗)rankf(xi,yj) 表示用𝑓(⋅,⋅)f(⋅,⋅) 对𝑌Y中的所有标签（注意是对𝑌Y中所有标签）进行降序排序，给个排名，最后返回的是𝑦𝑗yj标签在这个排序列表中的一个排名，排名越大，相关性越小。而 𝑚𝑎𝑥𝑦𝑗∈𝑦𝑖maxyj∈yi表示取到，真实标签𝑦𝑖yi中的标签在上面这个排名中最大的，那个排名。
如果真实标签𝑦𝑖yi被完全预测正确的话，取到的值是$\left | y^i \right | ，，y^i中的排名就是从1到中的排名就是从1到\left | y^i \right | 。如果。如果y^i中有一个标签中有一个标签y_j没有被预测正确，那么取的值就是那个标签没有被预测正确，那么取的值就是那个标签y_j在在Y中的排名，因为预测正确的那些都是排名最小（相关性最大）的那些标签，这个中的排名，因为预测正确的那些都是排名最小（相关性最大）的那些标签，这个y_j肯定是大于肯定是大于\left | y^i \right | $的。

1. Ranking Loss（度量的是：反序标签对的占比，也就是不相关标签比相关标签的相关性还要大的情况）





𝑟𝑙𝑜𝑠𝑠(𝑓)=1𝑝∑𝑖=1𝑝1∣∣𝑦𝑖∣∣∣∣𝑦𝑖⎯⎯⎯⎯⎯∣∣∣∣∣{(𝑦𝑗1,𝑦𝑗2)|𝑓(𝑥𝑖,𝑦𝑗1)≤𝑓(𝑥𝑖,𝑦𝑗2),(𝑦𝑗1,𝑦𝑗2)∈(𝑦𝑖×𝑦𝑖⎯⎯⎯⎯⎯)}∣∣∣rloss(f)=1p∑i=1p1|yi||yi¯||{(yj1,yj2)|f(xi,yj1)≤f(xi,yj2),(yj1,yj2)∈(yi×yi¯)}|



其中𝑦𝑖⎯⎯⎯⎯⎯yi¯为𝑦𝑖yi在𝑌Y上的补集。𝑦𝑗1yj1从相关的标签集𝑦𝑖yi中取，𝑦𝑗2yj2从不相关的标签集𝑦𝑖⎯⎯⎯⎯⎯yi¯中取，两两组合形成标签对。

1. Average Precision（度量的是：比特定标签更相关的那些标签的排名的占比）





𝑎𝑣𝑔𝑝𝑟𝑒𝑐(𝑓)=1𝑝∑𝑖=1𝑝1∣∣𝑦𝑖∣∣∑𝑦𝑗1∈𝑦𝑖∣∣{𝑦𝑗2|𝑟𝑎𝑛𝑘𝑓(𝑥𝑖,𝑦𝑗2)≤𝑟𝑎𝑛𝑘𝑓(𝑥𝑖,𝑦𝑗1),𝑦𝑗2∈𝑦𝑖}∣∣𝑟𝑎𝑛𝑘𝑓(𝑥𝑖,𝑦𝑗1)avgprec(f)=1p∑i=1p1|yi|∑yj1∈yi|{yj2|rankf(xi,yj2)≤rankf(xi,yj1),yj2∈yi}|rankf(xi,yj1)



**基于标签的评价指标**

1. Macro-averaging





𝐵𝑚𝑎𝑐𝑟𝑜(ℎ)=1𝑞∑𝑗=1𝑞𝐵(𝑇𝑃𝑗,𝐹𝑃𝑗,𝑇𝑁𝑗,𝐹𝑁𝑗)Bmacro(h)=1q∑j=1qB(TPj,FPj,TNj,FNj)



1. Micro-averaging





𝐵𝑚𝑖𝑐𝑟𝑜(ℎ)=𝐵(∑𝑗=1𝑞𝑇𝑃𝑗,∑𝑗=1𝑞𝐹𝑃𝑗,∑𝑗=1𝑞𝑇𝑁𝑗,∑𝑗=1𝑞𝐹𝑁𝑗)Bmicro(h)=B(∑j=1qTPj,∑j=1qFPj,∑j=1qTNj,∑j=1qFNj)



其中𝑇𝑃𝑗,𝐹𝑃𝑗,𝑇𝑁𝑗,𝐹𝑁𝑗TPj,FPj,TNj,FNj为单个标签下传统二分类的四个数量特征，真正例，假正例，真负例，假负例。
𝐵∈𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦,𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛,𝑅𝑒𝑐𝑎𝑙𝑙,𝐹𝛽B∈Accuracy,Precision,Recall,Fβ 表示对四个数量特征进行相关运算得到常规的二分类指标。
macro是先对单个标签下的数量特征计算得到常规指标，再对多个标签取平均。
micro是先对多个标签下的数量特征取平均，再根据数量特征计算得到常规指标。

1. AUC-macro（度量的是：“排序正确”的数据对的占比，macro是先对单个标签计算，再平均）
   （这里的“排序正确”指的是根据𝑓(⋅,⋅)f(⋅,⋅)函数，对于相关标签的打分会大于不相关标签的打分





𝐴𝑈𝐶𝑚𝑎𝑐𝑟𝑜=1𝑞∑𝑗=1𝑞∣∣{(𝑥′,𝑥″)|𝑓(𝑥′,𝑦𝑗)≥𝑓(𝑥″,𝑦𝑗),(𝑥′,𝑥″)∈𝑍𝑗×𝑍𝑗⎯⎯⎯⎯⎯⎯⎯}∣∣∣∣𝑍𝑗∣∣∣∣𝑍𝑗⎯⎯⎯⎯⎯⎯⎯∣∣AUCmacro=1q∑j=1q|{(x′,x″)|f(x′,yj)≥f(x″,yj),(x′,x″)∈Zj×Zj¯}||Zj||Zj¯|



其中𝑍𝑗={𝑥𝑖|𝑦𝑗∈𝑦𝑖,1≤𝑖≤𝑝}Zj={xi|yj∈yi,1≤i≤p}表示的是含有𝑦𝑗yj标签的样本数量
其中𝑍𝑗⎯⎯⎯⎯⎯⎯⎯={𝑥𝑖|𝑦𝑗∉𝑦𝑖,1≤𝑖≤𝑝}Zj¯={xi|yj∉yi,1≤i≤p}表示的是不含𝑦𝑗yj标签的样本数量。

1. AUC-micro（度量的是：“排序正确”的数据对的占比，micro是直接把多个标签考虑在内来计算占比）





𝐴𝑈𝐶𝑚𝑖𝑐𝑟𝑜=∣∣{(𝑥′,𝑥″,𝑦′,𝑦″)|𝑓(𝑥′,𝑦′)≥𝑓(𝑥″,𝑦″),(𝑥′,𝑦′)∈𝑆+,(𝑥″,𝑦″)∈𝑆−}∣∣|𝑆+||𝑆−|AUCmicro=|{(x′,x″,y′,y″)|f(x′,y′)≥f(x″,y″),(x′,y′)∈S+,(x″,y″)∈S−}||S+||S−|



其中𝑆+={(𝑥𝑖,𝑦𝑗)|𝑦𝑗∈𝑦𝑖,1≤𝑖≤𝑝}S+={(xi,yj)|yj∈yi,1≤i≤p}表示的是相关的样本标签对
其中𝑆−={(𝑥𝑖,𝑦𝑗)|𝑦𝑗∉𝑦𝑖,1≤𝑖≤𝑝}S−={(xi,yj)|yj∉yi,1≤i≤p}表示的是不相关的样本标签对

**学习算法**
可分为两类（具体算法如下图所示）

- 问题转换的方法：把多标签问题转为其它学习场景，比如转为二分类，标签排序，多分类
- 算法改编的方法：通过改编流行的学习算法去直接处理多标签数据，比如改编懒学习，决策树，核技巧。
  ![img](https://images2018.cnblogs.com/blog/1160281/201807/1160281-20180720101425425-720166607.png)

下面对图中的每个算法进行介绍。
**Binary Relevance**
把多个标签分离开来，对于q个标签，建立q个数据集和q个二分类器来进行预测。
这是最简单最直接的方法，是其它先进的多标签算法的基石。
没有考虑标签之间的关联性，是一个一阶策略（first-order）

**Classifier Chains**
首先按特定的顺序（这个顺序是自己决定的）对q个标签排个序，得到𝑦𝜏(1)≻𝑦𝜏(2)≻...≻𝑦𝜏(𝑞)yτ(1)≻yτ(2)≻...≻yτ(q)。对于第j个标签𝑦𝜏(𝑗)yτ(j)构建一个二分类的数据集





𝐷𝜏(𝑗)={([𝑥𝑖,𝑝𝑟𝑒𝑖𝜏(𝑗)],1{𝑦𝜏(𝑗)∈𝑦𝑖})|1≤𝑖≤𝑚}𝑤ℎ𝑒𝑟𝑒 𝑝𝑟𝑒𝑖𝜏(𝑗)=(1{𝑦𝜏(1)∈𝑦𝑖},...,1{𝑦𝜏(𝑗−1)∈𝑦𝑖})𝑇Dτ(j)={([xi,preτ(j)i],1{yτ(j)∈yi})|1≤i≤m}where preτ(j)i=(1{yτ(1)∈yi},...,1{yτ(j−1)∈yi})T



第j个标签构建的二分类数据集中，𝑥𝑖xi会concat上前j-1个标签值。
以这样chain式的方法构建q个数据集，训练q个分类器。
在预测阶段，由于第j个分类器需要用到前j-1个分类器预测出的标签集，所以需要顺序调用这q个分类器来预测。

1. 显然算法的好坏会受到顺序𝜏τ的影响，可以使用集成的方式，使用多个随机序列，对每个随机序列使用一部分的数据集进行训练。
2. 虽然该算法把问题分解成多个二分类，但由于它以随机的方式考虑了多个标签之间的关系，所以它是一个高阶策略（high-order）。
3. 该算法的一个缺点是丢失了平行计算的机会，因为它需要链式调用来进行预测

**Calibrated Label Ranking**
算法的基本思想是把多标签学习问题转为标签排序问题，该算法通过“成对比较”来实现标签间的排序。
对q个标签，可以构建q(q-1)/2个标签对，所以可以构建q(q-1)/2个数据集。





𝐷𝑗𝑘={(𝑥𝑖,𝜓(𝑦𝑖,𝑦𝑗,𝑦𝑘))|𝜙(𝑦𝑖,𝑦𝑗)≠𝜙(𝑦𝑖,𝑦𝑘),1≤𝑖≤𝑚}𝑤ℎ𝑒𝑟𝑒 𝜓(𝑦𝑖,𝑦𝑗,𝑦𝑘))={+1,−1,𝑖𝑓 𝜙(𝑦𝑖,𝑦𝑗)=+1 𝑎𝑛𝑑 𝜙(𝑦𝑖,𝑦𝑘)=−1𝑖𝑓 𝜙(𝑦𝑖,𝑦𝑗)=−1 𝑎𝑛𝑑 𝜙(𝑦𝑖,𝑦𝑘)=+1𝜙(𝑦𝑖,𝑦𝑗)={+1−1𝑖𝑓 𝑦𝑗∈𝑦𝑖𝑒𝑙𝑠𝑒Djk={(xi,ψ(yi,yj,yk))|ϕ(yi,yj)≠ϕ(yi,yk),1≤i≤m}where ψ(yi,yj,yk))={+1,if ϕ(yi,yj)=+1 and ϕ(yi,yk)=−1−1,if ϕ(yi,yj)=−1 and ϕ(yi,yk)=+1ϕ(yi,yj)={+1if yj∈yi−1else



1. 只有带有不同相关性的两个标签𝑦𝑗yj和𝑦𝑘yk的样本才会被包含在数据集𝐷𝑗𝑘Djk中，用该数据集训练一个分类器，当分类器返回大于0时，样本属于标签𝑦𝑗yj，否则属于标签𝑦𝑘yk。
2. 可以看到，每个样本𝑥𝑖xi会被包含在∣∣𝑦𝑖∣∣∣∣𝑦𝑖⎯⎯⎯⎯⎯∣∣|yi||yi¯|个分类器中。
3. 在预测阶段，根据分类器，每个样本和某个标签会产生一系列的投票，根据投票行为来做出最终预测。
4. 前面构造二分类器的方法使用one-vs-rest的方式，本算法使用one-vs-one，缓和类间不均衡的问题。
5. 缺点在于复杂性高，构建的分类器个数为q(q-1)/2，表现为二次增长。
6. 考虑两个标签之间的关联，是二阶策略（second-order）

**Random k-Labelsets**
算法的基本思想是把多标签学习问题转为多分类问题。把2𝑞2q个可能的标签集，映射成2𝑞2q个自然数。
映射函数记为𝜎𝑌σY，则原数据集变为𝐷+𝑌={(𝑥𝑖,𝜎𝑌(𝑦𝑖)) | 1≤𝑖≤𝑚}DY+={(xi,σY(yi)) | 1≤i≤m}。
所对应的新类别记为 $ \Gamma(D^+_Y) = { \sigma_Y(y^i) \ | \ 1 \leq i \leq m}，显然，显然 \left | \Gamma(D^+_Y) \right | \leq min(m, 2^{|Y|})$。
这样来训练一个多分类器，最后根据输出的自然数映射回标签集的算法称为LP（Label Powerest）算法，它有两个主要的局限性

1. LP预测的标签集是训练集中已经出现的，它没法泛化到未见过的标签集
2. 类别太大，低效

为了克服LP的局限性，Random k-Labelsets使用的LP分类器只训练Y中的一个长度为k的子集，然后集成大量的LP分类器来预测。
𝑌𝑘Yk表示𝑌Y的所有的长度为k的子集，𝑌𝑘(𝑙)Yk(l)表示随机取的一个长度为k的子集，这样就可以进行收缩样本空间，得到如下样本集和标签集。





𝐷+𝑌𝑘(𝑙)={(𝑥𝑖,𝜎𝑌𝑘(𝑙)(𝑦𝑖∩𝑌𝑘(𝑙))) | 1≤𝑖≤𝑚}DYk(l)+={(xi,σYk(l)(yi∩Yk(l))) | 1≤i≤m}







Γ(𝐷+𝑌𝑘(𝑙))={𝜎𝑌𝑘(𝑙)(𝑦𝑖∩𝑌𝑘(𝑙)) | 1≤𝑖≤𝑚}Γ(DYk(l)+)={σYk(l)(yi∩Yk(l)) | 1≤i≤m}



更进一步，我们随机取n个这样的子集：𝑌𝑘(𝑙𝑟),1≤𝑟≤𝑛Yk(lr),1≤r≤n来构造n个分类器做集成。
最后预测的时候需要计算两个指标，一个为标签j能达到的最大投票数，一个为实际投票数。





𝜏(𝑥,𝑦𝑗)=∑𝑟=1𝑛1{𝑦𝑗∈𝑌𝑘(𝑙𝑟)}τ(x,yj)=∑r=1n1{yj∈Yk(lr)}







𝜇(𝑥,𝑦𝑗)=∑𝑟=1𝑛1{𝑦𝑗∈𝜎−1𝑌𝑘(𝑙)(𝑔+𝑌𝑘(𝑙)(𝑥))}μ(x,yj)=∑r=1n1{yj∈σYk(l)−1(gYk(l)+(x))}



其中$ \sigma_{Yk(l)}{-1}(\cdot)表示从自然数映射回标签集的函数，表示从自然数映射回标签集的函数，g^+(\cdot)$表示分类器学习到的函数。最后预测的时以0.5为阈值进行预测，得到标签集。





𝑦={𝑦𝑗 | 𝜇(𝑥,𝑦𝑗) / 𝜏(𝑥,𝑦𝑗)>0.5 , 1≤𝑗≤𝑞}y={yj | μ(x,yj) / τ(x,yj)>0.5 , 1≤j≤q}



因为是随机长度为k的子集，考虑了多个标签之间的相关性，所以是高阶策略（high-order）。

**Multi-Label k-Nearest Neighbor（ML-KNN）**
用𝑁(𝑥)N(x)表示x的𝑘k个邻居，则𝐶𝑗=∑(𝑥,𝑦)∈𝑁(𝑥)1{𝑦𝑗∈𝑦}Cj=∑(x,y)∈N(x)1{yj∈y}表示样本x的邻居中带有标签𝑦𝑗yj的邻居个数。 用𝐻𝑗Hj表示样本x含有标签𝑦𝑗yj，根据**后验概率最大化**的规则，有





𝑦={𝑦𝑗 | 𝑃(𝐻𝑗 | 𝐶𝑗) / 𝑃(⌝𝐻𝑗 | 𝐶𝑗)>1 , 1≤𝑗≤𝑞}y={yj | P(Hj | Cj) / P(⌝Hj | Cj)>1 , 1≤j≤q}



根据**贝叶斯规则**，有





𝑃(𝐻𝑗 | 𝐶𝑗)𝑃(⌝𝐻𝑗 | 𝐶𝑗)=𝑃(𝐻)⋅𝑃(𝐶𝑗 | 𝐻𝑗)𝑃(⌝𝐻)⋅𝑃(𝐶𝑗 | 𝐻𝑗)P(Hj | Cj)P(⌝Hj | Cj)=P(H)⋅P(Cj | Hj)P(⌝H)⋅P(Cj | Hj)



**先验概率**𝑃(𝐻𝑗),𝑃(⌝𝐻𝑗)P(Hj),P(⌝Hj)可以通过训练集计算得到，表示样本带有或不带有标签𝑦𝑞yq的概率





𝑃(𝐻𝑗)=𝑠+∑𝑚𝑖=11{𝑦𝑗∈𝑦𝑖}𝑠×2+𝑚𝑃(⌝𝐻𝑗)=1−𝑃(𝐻𝑗) (1≤𝑗≤𝑞)P(Hj)=s+∑i=1m1{yj∈yi}s×2+mP(⌝Hj)=1−P(Hj) (1≤j≤q)



其中s是平滑因子，s为1时则使用的是拉普拉斯平滑。
条件概率的计算需要用到两个值





𝜅𝑗[𝑟]=∑𝑖=1𝑚1{𝑦𝑗∈𝑦𝑖}⋅1{𝛿𝑗(𝑥𝑖)=𝑟}  (0≤𝑟≤𝑘)𝜅̃𝑗[𝑟]=∑𝑖=1𝑚1{𝑦𝑗∉𝑦𝑖}⋅1{𝛿𝑗(𝑥𝑖)=𝑟}  (0≤𝑟≤𝑘)𝑤ℎ𝑒𝑟𝑒 𝛿𝑗(𝑥𝑖)=∑(𝑥∗,𝑦∗)∈𝑁(𝑥𝑖)1{𝑦𝑗∈𝑦∗}κj[r]=∑i=1m1{yj∈yi}⋅1{δj(xi)=r}  (0≤r≤k)κ~j[r]=∑i=1m1{yj∉yi}⋅1{δj(xi)=r}  (0≤r≤k)where δj(xi)=∑(x∗,y∗)∈N(xi)1{yj∈y∗}



𝜅𝑗[𝑟]κj[r]表示“含有标签𝑦𝑗yj而且r个邻居也含有标签𝑦𝑗yj的”样本的个数。
𝜅̃𝑗[𝑟]κ~j[r]表示“不含有标签𝑦𝑗yj但是r个邻居含有𝑦𝑗yj的”样本的个数。
根据这两个值，可以计算相应的**条件概率**





𝑃(𝐶𝑗 | 𝐻𝑗)=𝑠+𝜅𝑗[𝐶𝑗]𝑠×(𝑘+1)+∑𝑘𝑟=0𝜅𝑗[𝑟] (1≤𝑗≤𝑞,0≤𝐶𝑗≤𝑘)𝑃(𝐶𝑗 | ⌝𝐻𝑗)=𝑠+𝜅̃𝑗[𝐶𝑗]𝑠×(𝑘+1)+∑𝑘𝑟=0𝜅̃𝑗[𝑟] (1≤𝑗≤𝑞,0≤𝐶𝑗≤𝑘)P(Cj | Hj)=s+κj[Cj]s×(k+1)+∑r=0kκj[r] (1≤j≤q,0≤Cj≤k)P(Cj | ⌝Hj)=s+κ~j[Cj]s×(k+1)+∑r=0kκ~j[r] (1≤j≤q,0≤Cj≤k)



这两个条件概率表示的是，样本带有或不带有标签𝑦𝑗yj的条件下，它有𝐶𝑗Cj个邻居带有标签𝑦𝑗yj的概率。

1. 由上述的条件概率，先验概率则可以根据贝叶斯规则和后验概率最大化，计算出样本的标签集
2. 需要注意的是该方法不是KNN和独立二分类的简单结合，因为算法中还使用了贝叶斯来推理邻居信息
3. 没有考虑标签之间的相关性，是一阶策略（first-order）

**Multi-Label Decision Tree（ML-DT）**
使用决策树的思想来处理多标签数据，数据集T中，使用第l个特征，划分值为𝜗ϑ，计算出如下信息增益：





𝐼𝐺(𝑇,𝑙,𝜗)=𝑀𝐿𝐸𝑛𝑡(𝑇)−∑𝜌∈{−,+}|𝑇𝜌||𝑇|⋅𝑀𝐿𝐸𝑛𝑡(𝑇𝜌)𝑤ℎ𝑒𝑟𝑒 𝑇−={(𝑥𝑖,𝑦𝑖) | 𝑥𝑖𝑙≤𝑣,1≤𝑖≤𝑛}𝑤ℎ𝑒𝑟𝑒 𝑇+={(𝑥𝑖,𝑦𝑖) | 𝑥𝑖𝑙>𝑣,1≤𝑖≤𝑛}IG(T,l,ϑ)=MLEnt(T)−∑ρ∈{−,+}|Tρ||T|⋅MLEnt(Tρ)where T−={(xi,yi) | xil≤v,1≤i≤n}where T+={(xi,yi) | xil>v,1≤i≤n}



递归地构建一颗决策树，每次选取特征和划分值，使得上式的信息增益最大。
其中式子中的熵的公式可以按如下计算（为了方便计算，假定标签之间独立）。





𝑀𝐿𝐸𝑛𝑡(𝑇)=∑𝑗=1𝑞−𝑝𝑗𝑙𝑜𝑔2𝑝𝑗−(1−𝑝𝑗)𝑙𝑜𝑔2(1−𝑝𝑗)𝑤ℎ𝑒𝑟𝑒 𝑝𝑗=∑𝑛𝑖=11{𝑦𝑗∈𝑦𝑖}𝑛MLEnt(T)=∑j=1q−pjlog2pj−(1−pj)log2(1−pj)where pj=∑i=1n1{yj∈yi}n



1. 新样本到来时，向下遍历决策树的结点，找到叶子结点，若𝑝𝑗pj大于0.5则表示含有标签𝑦𝑗yj
2. 该算法不是决策树和独立二分类的简单结合（如果是的话，应该构建q棵决策树）
3. 没有考虑标签的相关性，是一阶策略（first-order）

**Ranking Support Vector Machine（Rank-SVM）**
使用最大间隔的思想来处理多标签数据。
Rank-SVM考虑系统对相关标签和不相关标签的排序能力。
考虑最小化𝑥𝑖xi到每一个“相关-不相关”标签对的超平面的距离，来得到间隔。





min(𝑥𝑖,𝑦𝑖)∈𝐷min(𝑦𝑗,𝑦𝑘)∈𝑦𝑖×𝑦𝑖⎯⎯⎯⎯⎯⟨𝑤𝑗−𝑤𝑘,𝑥𝑖⟩+𝑏𝑗−𝑏𝑘‖‖𝑤𝑗−𝑤𝑘‖‖min(xi,yi)∈Dmin(yj,yk)∈yi×yi¯⟨wj−wk,xi⟩+bj−bk‖wj−wk‖



像SVM一样对w和b进行缩放变换后可以对式子进行改写，然后最大化间隔，再调换分子分母进行改写，得到：





min𝑤𝑠𝑢𝑏𝑗𝑒𝑐𝑡 𝑡𝑜:max1≤𝑗<𝑘≤𝑞‖‖𝑤𝑗−𝑤𝑘‖‖2⟨𝑤𝑗−𝑤𝑘,𝑥𝑖⟩+𝑏𝑗−𝑏𝑘≥1(1≤𝑖≤𝑚, (𝑦𝑖,𝑦𝑘)∈𝑦𝑖×𝑦𝑖⎯⎯⎯⎯⎯)minwmax1≤j<k≤q‖wj−wk‖2subject to:⟨wj−wk,xi⟩+bj−bk≥1(1≤i≤m, (yi,yk)∈yi×yi¯)



为了简化，用sum操作来近似max操作





min𝑤𝑠𝑢𝑏𝑗𝑒𝑐𝑡 𝑡𝑜:∑𝑞𝑗=1‖‖𝑤𝑗‖‖2⟨𝑤𝑗−𝑤𝑘,𝑥𝑖⟩+𝑏𝑗−𝑏𝑘≥1(1≤𝑖≤𝑚, (𝑦𝑖,𝑦𝑘)∈𝑦𝑖×𝑦𝑖⎯⎯⎯⎯⎯)minw∑j=1q‖wj‖2subject to:⟨wj−wk,xi⟩+bj−bk≥1(1≤i≤m, (yi,yk)∈yi×yi¯)



跟SVM一样，为了软间隔最大化，引入松弛变量，得到下式：





min𝑤,Ξ𝑠𝑢𝑏𝑗𝑒𝑐𝑡 𝑡𝑜:∑𝑞𝑗=1‖‖𝑤𝑗‖‖2+𝐶∑𝑚𝑖=11∣∣𝑦𝑖∣∣∣∣𝑦𝑖⎯⎯⎯⎯⎯∣∣∑(𝑦𝑖,𝑦𝑘)∈𝑦𝑖×𝑦𝑖⎯⎯⎯⎯⎯)𝜉𝑖𝑗𝑘⟨𝑤𝑗−𝑤𝑘,𝑥𝑖⟩+𝑏𝑗−𝑏𝑘≥1−𝜉𝑖𝑗𝑘𝜉𝑖𝑗𝑘>0 (1≤𝑖≤𝑚, (𝑦𝑖,𝑦𝑘)∈𝑦𝑖×𝑦𝑖⎯⎯⎯⎯⎯)minw,Ξ∑j=1q‖wj‖2+C∑i=1m1|yi||yi¯|∑(yi,yk)∈yi×yi¯)ξijksubject to:⟨wj−wk,xi⟩+bj−bk≥1−ξijkξijk>0 (1≤i≤m, (yi,yk)∈yi×yi¯)



其中Ξ={𝜉𝑖𝑗𝑘 | 1≤𝑖≤𝑚, (𝑦𝑖,𝑦𝑘)∈𝑦𝑖×𝑦𝑖⎯⎯⎯⎯⎯}Ξ={ξijk | 1≤i≤m, (yi,yk)∈yi×yi¯}

1. 跟SVM一样，最终的式子是一个二次规划问题，通常调用现有的包来解。
2. 对于非线性问题则使用核技巧来解决。
3. 由于定义了”相关-不相关“标签对的超平面，这是个二阶策略（second-order）

**Collective Multi-Label Classifier（CML）**
该算法的核心思想最大熵原则。用(𝑥,𝑦),(x,y),表示任意的一个多标签样本，其中𝑦=(𝑦1,𝑦2,...,𝑦𝑞)∈{−1,+1}𝑞y=(y1,y2,...,yq)∈{−1,+1}q。
算法的任务等价于学习一个联合概率分布𝑝(𝑥,𝑦)p(x,y)，用𝐻𝑝(𝑥,𝑦)Hp(x,y)表示给定概率分布𝑝p时(𝑥,𝑦)(x,y)的信息熵。
最大熵原则认为熵最大的模型是最好的模型。





max𝑝𝐻𝑝(𝑥,𝑦)𝑠𝑢𝑏𝑗𝑒𝑐𝑡 𝑡𝑜:𝐸𝑝[𝑓𝑘(𝑥,𝑦)]=𝐹𝑘 (𝑘∈𝐾)maxpHp(x,y)subject to:Ep[fk(x,y)]=Fk (k∈K)



其中𝑓𝑘(𝑥,𝑦)fk(x,y)是一个特征函数，描述𝑥x和𝑦y之间的一个事实𝑘k，满足这个事实时返回1，否则返回0。
约束做的是希望这个分布上，特征函数的期望能够等于一个我们希望的值𝐹𝑘Fk，这个值通常通过训练集来估计。
解这个优化问题，会得到





𝑝(𝑦|𝑥)=1𝑍Λ(𝑥)𝑒𝑥𝑝(∑𝑘∈𝐾𝜆𝑘⋅𝑓𝑘(𝑥,𝑦))p(y|x)=1ZΛ(x)exp(∑k∈Kλk⋅fk(x,y))



其中Λ={𝜆𝑘|𝑘∈𝐾}Λ={λk|k∈K}表示一系列的权重。$Z_{\Lambda} = \sum_y exp(\sum_{k \in K} \lambda_k \cdot f_k(x,y)) 作为规范化因子。假设有一个高斯先验作为规范化因子。假设有一个高斯先验\lambda_k \sim N(0, \varepsilon^2)，就可以通过最大化以下这个𝑙𝑜𝑔后验概率来求得参数，就可以通过最大化以下这个log后验概率来求得参数\Lambda$。





𝑙(Λ|𝐷)=𝑙𝑜𝑔𝑃(𝐷|Λ)+𝑙𝑜𝑔𝑃(Λ)=𝑙𝑜𝑔∏(𝑥,𝑦)∈𝐷𝑝(𝑦|𝑥)+𝑙𝑜𝑔𝑃(Λ)=𝑙𝑜𝑔(∏(𝑥,𝑦)∈𝐷𝑝(𝑦|𝑥))−∑𝑘∈𝐾𝜆22𝜀2l(Λ|D)=logP(D|Λ)+logP(Λ)=log∏(x,y)∈Dp(y|x)+logP(Λ)=log(∏(x,y)∈Dp(y|x))−∑k∈Kλ22ε2



1. 这是个凸函数，可以调用现成的无约束优化方法比如BFGS直接求解。求得参数就可以得到要学习的概率分布𝑝(𝑦|𝑥)p(y|x)。
2. 对于一系列约束K，分为两个部分
3. 𝐾1={(𝑙,𝑗)|1≤𝑙≤𝑑,1≤𝑗≤𝑞}K1={(l,j)|1≤l≤d,1≤j≤q}，有𝑑⋅𝑞d⋅q个约束，特征函数为





𝑓𝑘(𝑥,𝑦)=𝑥𝑙⋅1{𝑦𝑗==1}, 𝑘=(𝑙,𝑗)∈𝐾1fk(x,y)=xl⋅1{yj==1}, k=(l,j)∈K1



1. 𝐾2=(𝑗1,𝑗2,𝑏1,𝑏2)|1≤𝑗1<𝑗2≤𝑞, 𝑏1,𝑏2∈−1,+1K2=(j1,j2,b1,b2)|1≤j1<j2≤q, b1,b2∈−1,+1，有4⋅(𝑞2)4⋅(q2)个约束，特征函数为





𝑓𝑘(𝑥,𝑦)=1{𝑦𝑗1=𝑏1}⋅1{𝑦𝑗2=𝑏2}, 𝑘=(𝑗1,𝑗2,𝑏1,𝑏2)∈𝐾2fk(x,y)=1{yj1=b1}⋅1{yj2=b2}, k=(j1,j2,b1,b2)∈K2



1. 由于K约束中考虑了标签对之间的关联，该算法是个二阶策略（second-order）。

**相关任务**

1. 多实例学习（Multi-instance learning）：每个样本由多个实例和一个标签组成，多个实例中至少一个为正，认为该样本为正。和多标签学习的输出空间模糊相反，多实例学习是输入空间模糊。
2. 有序分类（Ordinal classification）：对于每个标签，不再是简单地判断是还是否，而是改成一系列的等级排序，把𝑦𝑗={−1,+1}yj={−1,+1}替换成𝑦𝑗={𝑚1,𝑚2,...,𝑚𝑘}, 𝑤ℎ𝑒𝑟𝑒 𝑚1<𝑚2<...<𝑚𝑘yj={m1,m2,...,mk}, where m1<m2<...<mk
3. 多任务学习（Multi-task learning）：同时训练多个任务，相关任务之间的训练信息会帮助其它任务。比如目标定位既要识别有没有目标（分类问题）又要定位出目标的位置（回归问题）。
4. 数据流学习（Data streams classification）：真实世界的目标是在线生成和实时产生的，如何处理这些数据就是数据流学习要做的事。一个关键的挑战就是“概念漂移”（目标变量的统计特性随着时间的推移以不可预见的方式变化），一般处理方式有：当一大批新数据到来时更新分类器；维持一个检测器来警惕概念漂移；假定过去数据的影响会随着时间而衰减。

**总结**

1. 论文主要介绍了多标签学习的一些概念定义，策略，评价指标，以及8个有代表性的算法，其中对多种评价指标和多个算法都做了清晰的分类和详细的阐述。
2. 尽管挖掘标签关联性的想法被应用到许多算法中，但是仍然没有一个正式的机制。有研究表示多标签之间的关联可能是非对称的（我对你的影响和你对我的影响是不同的），局部的（不同样本之间的标签相关性不同，很少关联性是所有样本都满足的）。
3. 但是不管怎么说，充分理解和挖掘标签之间的相关性，是多标签学习的法宝。尤其是巨大输出空间场景下。