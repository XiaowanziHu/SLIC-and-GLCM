#include "stdafx.h"
#include <iostream>
#include <time.h>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "SLIC.h"

using namespace std;
using namespace cv;

#define Width 935;//ͼ����

#define GLCM_DIS 3  //�Ҷȹ��������ͳ�ƾ���  
#define GLCM_CLASS 16 //����Ҷȹ��������ͼ��Ҷ�ֵ�ȼ��� 

//����Ҷȹ��������ĸ�����
typedef enum GLCM_ANGLE
{
	GLCM_ANGLE_HORIZATION,
	GLCM_ANGLE_VERTICAL,
	GLCM_ANGLE_DIGONAL_45,
	GLCM_ANGLE_DIGONAL_135,
}GLCM_ANGLE;

int imgOpenCV2SLIC(Mat img, int &height, int &width, int &dim, unsigned int * &image);
int imgSLIC2openCV(unsigned int *image, int height, int width, int dim, Mat &imgSLIC);
void creatAlphaMat(Mat &mat);
void ReadKlabel(string path, string pathOut, const int*klabels);
int CalGlCM(int* pImage, GLCM_ANGLE angleDirection, double* featureVector, const int* klabels, int width, int height, int kl);

//�����ѡȡѵ������
IplImage* src = 0;
void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);//����ṹ��ʼ��  

	ofstream pointfile1, pointfile2;
	pointfile1.open("examplepoint-daolu-1000", ios::binary | ios::app | ios::in | ios::out);
	pointfile2.open("examplepoint-daolu-1000-result", ios::binary | ios::app | ios::in | ios::out);

	if ((event == CV_EVENT_LBUTTONDOWN) && (flags))//�����������¼�����  
	{
		int result = 0;
		CvPoint pt = cvPoint(x, y);//��ȡ��ǰ��ĺ�������ֵ  
		char temp[16];
		sprintf_s(temp, "(%d,%d)", pt.y, pt.x);//��ӡ��ǰ����ֵ  
		//cvPutText(src, temp, pt, &font, cvScalar(0, 0, 255, 0)); //��ͼ���д�ӡ��ǰ����ֵ   
		cvCircle(src, pt, 2, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);//����ͼ��ǰ������»�Բ  
		cvShowImage("src", src);
		//��������
		result = pt.y * Width;
		result += pt.x;
		pointfile1 << pt.y << " " << pt.x << "\n";
		pointfile2 << result << "\n";
	}
}

//OpenCV Matͼ������ת��ΪSLICͼ������
int imgOpenCV2SLIC(Mat img, int &height, int &width, int &dim, unsigned int * &image)
{
	int error = 0;
	if (img.empty()) //��һ������Ƿ�ɹ���ͼ 
	{
		error = 1;
	}

	dim = img.channels();//ͼ��ͨ����Ŀ
	height = img.rows;
	width = img.cols;

	int imgSize = width*height;

	unsigned char *pImage = new unsigned char[imgSize * 4];
	if (dim == 1)
	{
		for (int j = 0; j < height; j++)
		{
			uchar * ptr = img.ptr<uchar>(j);
			for (int i = 0; i < width; i++)
			{
				pImage[j * width * 4 + 4 * i + 3] = 0;
				pImage[j * width * 4 + 4 * i + 2] = ptr[0];
				pImage[j * width * 4 + 4 * i + 1] = ptr[0];
				pImage[j * width * 4 + 4 * i] = ptr[0];
				ptr++;
			}
		}
	}
	else if (dim == 3)
	{
		for (int j = 0; j < height; j++)
		{
			Vec3b * ptr = img.ptr<Vec3b>(j);
			for (int i = 0; i < width; i++)
			{
				pImage[j * width * 4 + 4 * i + 3] = 0;
				pImage[j * width * 4 + 4 * i + 2] = ptr[0][2];//R
				pImage[j * width * 4 + 4 * i + 1] = ptr[0][1];//G
				pImage[j * width * 4 + 4 * i] = ptr[0][0];//B        
				ptr++;
			}
		}
	}

	else  error = 1;

	image = new unsigned int[imgSize];
	memcpy(image, (unsigned int*)pImage, imgSize*sizeof(unsigned int));
	delete pImage;

	return error;

}

//SLICͼ������ת��ΪOpenCV Matͼ������
int imgSLIC2openCV(unsigned int *image, int height, int width, int dim, Mat &imgSLIC)
{
	int error = 0;//ת���Ƿ�ɹ��ı�־���ɹ�Ϊ0��ʶ��Ϊ1

	if (dim == 1)
	{
		imgSLIC.create(height, width, CV_8UC1);
		//�����������أ�����������ֵ 
		for (int j = 0; j< height; ++j)
		{
			//��ȡ�� i��������ָ�� 
			uchar * p = imgSLIC.ptr<uchar>(j);
			//�Ե� i�е�ÿ������(byte)���� 
			for (int i = 0; i < width; ++i)
				p[i] = (unsigned char)(image[j*width + i] & 0xFF);
		}
	}

	else if (dim == 3)
	{
		imgSLIC.create(height, width, CV_8UC3);
		//�����������أ�����������ֵ 
		for (int j = 0; j < height; ++j)
		{
			//��ȡ�� i��������ָ�� 
			Vec3b * p = imgSLIC.ptr<Vec3b>(j);
			for (int i = 0; i < width; ++i)
			{
				p[i][0] = (unsigned char)(image[j*width + i] & 0xFF); //Blue 
				p[i][1] = (unsigned char)((image[j*width + i] >> 8) & 0xFF); //Green 
				p[i][2] = (unsigned char)((image[j*width + i] >> 16) & 0xFF); //Red 
			}
		}
	}

	else  error = 1;

	return error;
}

//����ѵ��������ǩ
//pathIn����ѵ����������(x * width + y), pathOut���ѵ��������ǩ
void ReadKlabel(string pathIn, string pathOut, const int* klabels)
{
	string line;
	int num = 0;
	int i = 0;
	ifstream infile;
	infile.open(pathIn, ios::binary | ios::app | ios::in | ios::out);

	ofstream outfile;
	outfile.open(pathOut, ios::binary | ios::app | ios::in | ios::out);
	while (getline(infile, line))
	{
		istringstream stream(line);
		int x;
		while (stream >> x)
		{
			//cout << x << endl;
			outfile << klabels[x] << "\r\n";
		}
	}
	cout << "read over!" << endl;
}

//��ȡͼ������ֵ
//pImage�Ҷ�ͼ�� featureVector�������ֵ
int CalGlCM(int* pImage, GLCM_ANGLE angleDirection, double* featureVector, const int* klabels, int width, int height, int kl)
{
	int i, j;
	if (NULL == pImage)
		return 1;

	float * glcm = new float[GLCM_CLASS * GLCM_CLASS];
	int * histImage = new int[width * height];
	int imgsize = width * height;

	if (NULL == glcm || NULL == histImage)
		return 2;

	//�Ҷȵȼ���---��GLCM_CLASS���ȼ�(ѹ��ͼ��)
	for (i = 0; i < height; i++)
	for (j = 0; j < width; j++)
		histImage[i * width + j] = (int)(pImage[i * width + j] * GLCM_CLASS / 256);

	//��ʼ����������
	for (i = 0; i < GLCM_CLASS; i++)
	for (j = 0; j < GLCM_CLASS; j++)
		glcm[i * GLCM_CLASS + j] = 0;

	//����Ҷȹ�������(������ͼ���)
	int w, k, l;
	int kl0 = -1, kl1 = -1, kl2 = -1;

	//ˮƽ����
	if (angleDirection == GLCM_ANGLE_HORIZATION)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				l = histImage[i * width + j];
				kl0 = klabels[i * width + j];
				if (kl0 != kl) continue;
				else
				{
					if ((i * width + j + GLCM_DIS) <(width * height))
						kl1 = klabels[i * width + j + GLCM_DIS];
					else kl1 = -1;  //�����ڷָ���ڣ���ֵ-1��������ֵΪ-1����kl1������һ�ε�ֵ��
					if ((i * width + j - GLCM_DIS) >= 0)
						kl2 = klabels[i * width + j - GLCM_DIS];
					else kl2 = -1;

					if (j + GLCM_DIS >= 0 && j + GLCM_DIS < width && kl1 == kl)
					{
						k = histImage[i* width + j + GLCM_DIS];
						glcm[l * GLCM_CLASS + k]++;
					}

					if (j - GLCM_DIS >= 0 && j - GLCM_DIS < width && kl2 == kl)
					{
						k = histImage[i* width + j - GLCM_DIS];
						glcm[l * GLCM_CLASS + k]++;
					}
				}
			}
		}
	}
	//��ֱ����
	else if (angleDirection == GLCM_ANGLE_VERTICAL)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				l = histImage[i * width + j];
				kl0 = klabels[i * width + j];
				if (kl0 != kl) continue;
				else
				{
					if ((i + GLCM_DIS) * width + j < width * height)
						kl1 = klabels[(i + GLCM_DIS) * width + j];
					else kl1 = -1;
					if ((i - GLCM_DIS) * width + j >= 0)
						kl2 = klabels[(i - GLCM_DIS) * width + j];
					else kl2 = -1;

					if (i + GLCM_DIS >= 0 && i + GLCM_DIS < width && kl1 == kl)
					{
						k = histImage[(i + GLCM_DIS) * width + j];
						glcm[l * GLCM_CLASS + k]++;
					}
					if (i - GLCM_DIS >= 0 && i - GLCM_DIS < width && kl2 == kl)
					{
						k = histImage[(i - GLCM_DIS) * width + j];
						glcm[l * GLCM_CLASS + k]++;
					}
				}
			}
		}
	}
	//�ԽǷ���
	else if (angleDirection == GLCM_ANGLE_DIGONAL_135)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				l = histImage[i * width + j];
				kl0 = klabels[i * width + j];
				if (kl0 != kl) continue;
				else
				{
					if ((i + GLCM_DIS) * width + j + GLCM_DIS < width * height)
						kl1 = klabels[(i + GLCM_DIS) * width + j + GLCM_DIS];
					else kl1 = -1;
					if ((i - GLCM_DIS) * width + j - GLCM_DIS >= 0)
						kl2 = klabels[(i - GLCM_DIS) * width + j - GLCM_DIS];
					else kl2 = -1;

					if (i + GLCM_DIS >= 0 && i + GLCM_DIS < height && j + GLCM_DIS >= 0 && j + GLCM_DIS < width && kl1 == kl)
					{
						k = histImage[(i + GLCM_DIS) * width + j + GLCM_DIS];
						glcm[l * GLCM_CLASS + k]++;
					}
					if (i - GLCM_DIS >= 0 && i - GLCM_DIS < height && j - GLCM_DIS >= 0 && j - GLCM_DIS < width && kl2 == kl)
					{
						k = histImage[(i - GLCM_DIS) * width + j - GLCM_DIS];
						glcm[l * GLCM_CLASS + k]++;
					}
				}
			}
		}
	}
	else if (angleDirection == GLCM_ANGLE_DIGONAL_45)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				l = histImage[i * width + j];
				kl0 = klabels[i * width + j];
				if (kl0 != kl) continue;
				else
				{
					if ((i - GLCM_DIS) * width + j + GLCM_DIS >= 0)
						kl1 = klabels[(i - GLCM_DIS) * width + j + GLCM_DIS];
					else kl1 = -1;
					if ((i + GLCM_DIS) * width + j - GLCM_DIS < width * height)
						kl2 = klabels[(i + GLCM_DIS) * width + j - GLCM_DIS];
					else kl2 = -1;

					if (i - GLCM_DIS >= 0 && i - GLCM_DIS < height && j + GLCM_DIS >= 0 && j + GLCM_DIS < width && kl1 == kl)
					{
						k = histImage[(i - GLCM_DIS) * width + j + GLCM_DIS];
						glcm[l * GLCM_CLASS + k]++;
					}
					if (i + GLCM_DIS >= 0 && i + GLCM_DIS < height && j - GLCM_DIS >= 0 && j - GLCM_DIS < width && kl2 == kl)
					{
						k = histImage[(i + GLCM_DIS) * width + j - GLCM_DIS];
						glcm[l * GLCM_CLASS + k]++;
					}
				}
			}
		}
	}

	//��һ��
	float total = 0;
	for (i = 0; i < GLCM_CLASS; i++)
	for (j = 0; j < GLCM_CLASS; j++)
		total += glcm[i * GLCM_CLASS + j];

	for (i = 0; i < GLCM_CLASS; i++)
	for (j = 0; j < GLCM_CLASS; j++)
		glcm[i * GLCM_CLASS + j] /= total;

	//��������ֵ
	double entropy = 0, energy = 0, contrast = 0, homogenity = 0;
	for (i = 0; i < GLCM_CLASS; i++)
	{
		for (j = 0; j < GLCM_CLASS; j++)
		{
			//��
			if (glcm[i * GLCM_CLASS + j] > 0)
				entropy -= glcm[i * GLCM_CLASS + j] * log10(double(glcm[i * GLCM_CLASS + j]));
			//����
			energy += glcm[i * GLCM_CLASS + j] * glcm[i * GLCM_CLASS + j];
			//�Աȶ�
			contrast += (i - j) * (i - j) * glcm[i * GLCM_CLASS + j];
			//һ����
			homogenity += 1.0 / (1 + (i - j) * (i - j)) * glcm[i * GLCM_CLASS + j];
		}
	}
	//��������ֵ
	i = 0;
	featureVector[i++] = entropy;
	featureVector[i++] = energy;
	featureVector[i++] = contrast;
	featureVector[i++] = homogenity;

	delete[] glcm;
	delete[] histImage;
	return 0;
}

int main()
{
	Mat imgRGB;
	time_t tStart, tEnd, exeT;
	
	imgRGB = imread("yangben1.tif");
	if (imgRGB.empty() == true){ cout << "can not open rgb image!" << endl;}
	
	unsigned int *image;	
	int height, width, dim;
	long imgSize;
	int numlabels(0);

	imgOpenCV2SLIC(imgRGB, height, width, dim, image);
	imgSize = height* width;
	
	tStart = clock();

	int k = 10000;// ����ĳ���������
	double m = 30;//Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
	int* klabels = nullptr;
	if (0 == klabels) klabels = new int[imgSize];
	//const int kstep = sqrt(double(imgSize) / double(k));

	// Perform SLIC on the image buffer
	SLIC segment;
	segment.PerformSLICO_ForGivenK(image, width, height, klabels, numlabels, k, m);
	//Alternately one can also use the function PerformSLICO_ForGivenStepSize() for a desired superpixel size
	//segment.PerformSLICO_ForGivenStepSize(labels, width, height, klabels, numlabels, kstep, m);

	//����ǩ���浽�ı��ļ���
	/*string filename = "1.jpg";
	string savepath = "yourpathname1";
	segment.SaveSuperpixelLabels(klabels, width, height, filename, savepath);*/

	// ���Ʊ߽�
	segment.DrawContoursAroundSegments(image, klabels, width, height, 0xff0000);

    //��ʼ��ǩ����
	ofstream outfile;
	outfile.open("klabels1-10000", ios::binary | ios::app | ios::in | ios::out);
	for (int j = 0; j < height; ++j)
	{
		//��ȡ�� i��������ָ�� 
		for (int i = 0; i < width; ++i)
		{
			int index = j*width + i;
			outfile << klabels[index];
			outfile << "\r";
			outfile << "\n";
		}
	}
	
	Mat ReImgSLIC;
	imgSLIC2openCV(image, height, width, dim, ReImgSLIC);

	if (image) delete[] image;  // Clean up
	
	tEnd = clock();
	exeT = tEnd - tStart;

	//RGB2Gray ��������ֵ�������������gray��
	Mat imgGray;
	cvtColor(imgRGB, imgGray, CV_BGR2GRAY);
	int *gray = new int[imgSize];
	for (int i = 0; i < height; i++)
	for (int j = 0; j < width;j++)
	{
		gray[i * width + j] = (int)imgGray.at<uchar>(i,j);
	}
	//����ָ�������ֵ
	double* featureVector = new double[4];
	ofstream featureFile;
	featureFile.open("featureFile-all-10000.txt", ios::binary | ios::app | ios::in | ios::out);
	for (int kl = 0; kl < k; kl++)    //�ñ�ǩ����ÿһ���ָ��
	{
		float entropy = 0, energy = 0, contrast = 0,homogenity = 0;
		CalGlCM(gray, GLCM_ANGLE_HORIZATION, featureVector, klabels, width, height, kl);
		entropy += featureVector[0]; energy += featureVector[1]; contrast += featureVector[2]; homogenity += featureVector[3];

		CalGlCM(gray, GLCM_ANGLE_VERTICAL, featureVector, klabels, width, height, kl);
		entropy += featureVector[0]; energy += featureVector[1]; contrast += featureVector[2]; homogenity += featureVector[3];

		CalGlCM(gray, GLCM_ANGLE_DIGONAL_45, featureVector, klabels, width, height, kl);
		entropy += featureVector[0]; energy += featureVector[1]; contrast += featureVector[2]; homogenity += featureVector[3];

		CalGlCM(gray, GLCM_ANGLE_DIGONAL_135, featureVector, klabels, width, height, kl);
		entropy += featureVector[0]; energy += featureVector[1]; contrast += featureVector[2]; homogenity += featureVector[3];

		entropy /= 4, energy /= 4, contrast /= 4, homogenity /= 4;
		featureFile << entropy << " " << energy << " " << contrast << " " << homogenity << "\r\n";
	}
	if (featureVector) delete[] featureVector;

	//��ȡѵ����������
	/*src = cvLoadImage("SLIC Segmentation1-10000.jpg", 1); //����ͼ��
	cvNamedWindow("src", 1);//�½�����  
	cvSetMouseCallback("src", on_mouse, 0);  //ע�������Ӧ�ص����� 
	cvShowImage("src", src);
	cvWaitKey(0);
	cvDestroyAllWindows();//�������д���  
	cvReleaseImage(&src);//�ͷ�ͼ������ 

	//��ȡ��Ӧѵ��������ǩ
	ReadKlabel("examplepoint-daolu-10000-result","klabels-10000-daolu", klabels);*/

	if (klabels) delete[] klabels;
	//if (gray) delete[] gray;

	//�����ʾ
    /*cout << "SLICִ��ʱ��exeT��" << exeT << "����" << endl;
	imshow("imgSLIC", ReImgSLIC);*/
	imwrite("SLIC Segmentation1-10000.jpg", ReImgSLIC);
	cout << "over!";
	getchar();
	return 0;
	}



