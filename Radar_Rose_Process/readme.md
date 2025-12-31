运行逻辑如下：

评估前置处理：

1.先运行circle_type 进行网格加密

网格加密的密度由interval调控

2.然后运行data_solve剔除误差较大的图片，获得初步的evaluation_datasets.json ，其中data_solve中origin_interval_count是原图片的密度，interval与circle_type相同

3.之后运行axis_find_type，进行坐标轴寻找，获取包含坐标轴的evaluation_datasets

4.运行filter_datasets_with....，筛选包含坐标轴的evaluation_datasets_with_axis.json 为最终评估json

开始评估:

修改evalution_type中对应路径，改成json位置，然后运行评估，结果会保存为coordinates_by_image_radar_gemini-2.0-flash.json类似格式

data文件夹长这样，circle储存原数据，out储存输出数据amp和feedback储存中间图片

without grid 为消融实验部分

先运行resultsbefore 在运行resultslater 查看评估结果表格
