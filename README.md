# 案例介绍

本案例通过对多导睡眠图（Polysomnography,PSG）数据进行睡眠阶段的分类来判断睡眠类型。
训练：对GroupA的睡眠数据进行训练测试：利用训练结果对GroupB的睡眠数据进行测试，判断其睡眠类型。



# 案例步骤

## （一）导入数据库

本案例用的数据是来自于PhysioNet上关于健康受试者的年龄对睡眠影响研究的公开数据集的一个子集。

mne.datasets.sleep_physionet.age.fetch_data可以下载PhysioNet数据集的子数据集。

该子数据集中有四个edf文件，分别是SC4001E0-PSG、SC4011E0-PSG、SC4001EC-Hypnogram、SC4011EH-Hypnogram。其中前两个是包含PSG多导睡眠图总计约40小时的数据文件，后两个是与前两个对应的包含专家记录的注释文件。![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps1.jpg)

##  

## **（二）加载数据**

从两个edf文件中加载数据，最终目标是获得时间片段(epochs)，然后，将这两个对象合并到mne.io.Raw对象中，就可以根据注释的描述提取事件以获得时间片段(epochs)

subjects：表示想要使用哪些受试者对象，可供选择的受试者对象范围为0-19。

recording：表示夜间记录的编号(索引)，有效值为：[1]、[2]或[1、2]。

以GROUPA代表实验组，GROUPB代表测试组。具体代码如下。

 

![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps2.jpg)

![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps3.jpg) 



 

本实验采用AASM的标准，使用5个阶段：唤醒（W），阶段1（N1期），阶段2（N2期），阶段3/4（N3期）和REM睡眠(R)。为此，这里使用mne.events_from_annotations()中的event_id参数来选择感兴趣的事件，并将事件标识符与每个事件相关联。



![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps4.jpg) 



 

![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps5.jpg) 

 



 

根据注释中的事件从数据创建epochs(时间片段)

![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps6.jpg) 

创建了8406个时间片段，其中属于N1期的有327个，N2期的有1686个，N3期的315个，REM期510个，W期5568个。

 

![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps7.jpg) 



## **（三）加载测试组的数据作为测试数据**

### 1获取数据

参照上文中的方法来获取测试组的测试数据。



![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps8.jpg) 

### 2特征工程

观察不同睡眠阶段的功率谱密度(PSD)图，可以看到不同睡眠阶段具有不同的特征。这些签名在实验组和测试组的数据中保持相似。在本实验的剩余部分中，将基于特定频带中的相对功率来创建EEG特征，以捕获数据中睡眠阶段之间的差异。



![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps9.jpg) 

  ![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps10.jpg)



## **（四）设计scikit-learn转换器**

创建一个函数，根据特定频带中的相对功率提取脑电图特征，从而能够根据脑电图信号预测睡眠阶段。

脑电相对功率带特征提取该函数接受一个""mne.Epochs"对象，并基于与scikit-learn兼容的特定频带中的相对功率创建EEG特征。

![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps11.jpg) 

上述特征提取中delta、theta、alpha、sigam、beta是脑电波中有关睡眠分期的指标，具体信息参考下图。

![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps12.jpg) 



## （五）根据实验组的数据来预测测试组的睡眠阶段

使用scikit-learn进行多分类，下面展示了解决如何从实验组的数据中预测测试组的睡眠阶段并尽可能避免重复样板代码的问题。这里将利用sckit-learn的Pipeline和FunctionTransformer。Pipeline可以将许多算法模型串联起来，可以用于把多个estamitors级联成一个estamitor,比如将特征提取、归一化、分类组织在一起形成一个典型的机器学习问题工作流。FunctionTransformer将python函数转换为与estamitor兼容的对象。

![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps13.jpg) 

准确率如下图示。

![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps14.jpg) 

预测的准确精度为84.1%

查看分类报告做进一步分析

 

![img](file:///C:\Users\19029\AppData\Local\Temp\ksohtml3616\wps15.jpg) 

从分类报告中可以看出，测试组的每个阶段训练测试样本，以及对应的睡眠阶段的精度。比如W阶段的精度为86%,测试样本为5568。测试总样本为8406。也可以看到其他一些指标比如召回率和F1值。