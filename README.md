# SLIC-and-GLCM
对超像素分割之后的不规则分割块进行纹理特征提取，并按照标签顺序保存到文本中


首先对图像进行超像素分割，得到分割图像以及每一像素点所属标签；
然后对每一超像元(即分割块)进行纹理特征提取，由于超像元是不规则的，且数目较多，面积太小，故直接单独对超像元进行特征提取误差太大；
对原始图像，利用标签，计算出每一超像元的灰度共生矩阵，最终得到每一超像元的纹理特征。
计算灰度共生矩阵具体实施：
第k个超像元内所有像素点的标签均为k，
遍历整幅图像，找到第一个标签为k的像素点
按照特定的规则找第二个像素点，并判断这个像素点的标签是否为k，若为k，则计入灰度共生矩阵中
按照以上方法找到n个超像元的灰度共生矩阵，最后得到纹理特征
