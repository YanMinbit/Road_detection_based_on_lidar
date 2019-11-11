//官方例子
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

int main( int /*argc*/, char** /*argv*/ )
{
    const int N = 4;
    const int N1 = (int)sqrt((double)N);//平方根
    const Scalar colors[] =
    {
        Scalar(0,0,255), Scalar(0,255,0),
        Scalar(0,255,255),Scalar(255,255,0)
    };

    int i, j;
    int nsamples = 100;
    Mat samples( nsamples, 2, CV_32FC1 );//100行，2列，单通道 总的样本
    Mat labels;
    Mat img = Mat::zeros( Size( 500, 500 ), CV_8UC3 );
    Mat sample( 1, 2, CV_32FC1 );//1行，2列，单通道 单个样本
    CvEM em_model;//注意这个的官网对应的是https://docs.opencv.org/ref/2.4/d0/da8/classCvEM.html，但是不好理解，
    //参考https://docs.opencv.org/3.1.0/d1/dfb/classcv_1_1ml_1_1EM.html#a2d02b45a574d51a72263e9c53cdc4f09
    //或者https://docs.opencv.org/2.4.8/modules/ml/doc/expectation_maximization.html#em-train
    //比较好理解，后面两个对应opencv 3.0和2.0，用法和解释差不多
    CvEMParams params;

    samples = samples.reshape(2, 0);//只是在逻辑上改变矩阵的行列数或者通道数，没有任何的数据的复制，也不会增减任何数据，因此这是一个O(1)的操作，它要求矩阵是连续的。 cn：目标通道数，如果是0则保持和原通道数一致；    rows：目标行数，同上是0则保持不变； 改变后的矩阵要满足 rows*cols*channels  跟原数组相等
   // Mat::Resize( )  是改变矩阵的行数，会引起矩阵的重新分配。
    //cv::resize( ) 这个是通过插值的方式来改变图像的尺寸，貌似不支持int型的元素，uchar，float和double都可以。
    //https://blog.csdn.net/monologue_/article/details/8659632 这里是上面一个很好的区分的例子
    for( i = 0; i < N; i++ )
    {
        // form the training samples
        Mat samples_part = samples.rowRange(i*nsamples/N, (i+1)*nsamples/N );

        Scalar mean(((i%N1)+1)*img.rows/(N1+1),
                    ((i/N1)+1)*img.rows/(N1+1));
        Scalar sigma(30,30);
        randn( samples_part, mean, sigma );
     /*   Parameters:

            dst – output array of random numbers; the array must be pre-allocated and have 1 to 4 channels.
            mean – mean value (expectation) of the generated random numbers.
            stddev – standard deviation of the generated random numbers; it can be either a vector (in which case a diagonal standard deviation matrix is assumed) or a square matrix.
*/
    }

 /*
    void randShuffle(InputOutputArray dst, double iterFactor=1., RNG* rng=0 ):
    Parameters:

        dst – input/output numerical 1D array.
        iterFactor – scale factor that determines the number of random swap operations (see the details below).
        rng – optional random number generator used for shuffling; if it is zero, theRNG() () is used instead.

The function randShuffle shuffles the specified 1D array by randomly choosing pairs of elements and swapping them. The number of such swap operations will be dst.rows*dst.cols*iterFactor .
    */

    samples = samples.reshape(1, 0);//列数是根据总的数据量不变自动得到

    // initialize model parameters
    params.covs      = NULL;
    params.means     = NULL;
    params.weights   = NULL; //混合高斯的权值
    params.probs     = NULL;
    params.nclusters = N;   //聚类个数
    params.cov_mat_type       = CvEM::COV_MAT_SPHERICAL;
    params.start_step         = CvEM::START_AUTO_STEP;
    params.term_crit.max_iter = 300;
    params.term_crit.epsilon  = 0.1;
    params.term_crit.type     = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;






    // cluster the data
    em_model.train( samples, Mat(), params, &labels );

    Mat Probs;
    Probs = em_model.getProbs();//每个样本属于每个类的概率
    double Likelihood = em_model.getLikelihood();//我：总样本导致最终分布，在这种分布下样本出现的概率
    //std::cout<<"Probs:"<<Probs<<std::endl;//nsamples＊4的矩阵，0到1之间，一行和为1
    std::cout<<"Likelihood:"<<Likelihood<<std::endl;//Likelihood:-760.93




    	Mat Weights;
    	Mat Means;//数据是二维的
    	std::vector<Mat> Covs;//每个高斯有一个协方差矩阵

    	Weights = em_model.getWeights();
    	Means = em_model.getMeans();
    	em_model.getCovs(Covs);

    	std::cout<<"Weights:"<<Weights<<std::endl;
    	std::cout<<"Means:"<<Means<<std::endl;
    	std::cout<<"Covs[0]:"<<Covs[0]<<std::endl;
    	std::cout<<"Covs[1]:"<<Covs[1]<<std::endl;
    	std::cout<<"Covs[2]:"<<Covs[2]<<std::endl;
    	std::cout<<"Covs[3]:"<<Covs[3]<<std::endl;

    	/*
    	 * 一种输出：
    	 *Weights:[0.2484987579776698, 0.2511391910133878, 0.2503694395787522, 0.2499926114301902]
  Means:[344.0927627467929, 166.9016833028004;
  340.0973074730531, 329.6253407169998;
  155.3868576605689, 330.5971522654546;
  164.8886348316039, 162.8848468835046]
Covs[0]:[855.8041512134523, 0;
  0, 855.8041512134523]
Covs[1]:[946.5831238281161, 0;
  0, 946.5831238281161]
Covs[2]:[836.2617049305286, 0;
  0, 836.2617049305286]
Covs[3]:[963.2737497469204, 0;
  0, 963.2737497469204]
    	 *
    	 *
    	 */

#if 0
    // the piece of code shows how to repeatedly optimize the model
    // with less-constrained parameters
    //(COV_MAT_DIAGONAL instead of COV_MAT_SPHERICAL)
    // when the output of the first stage is used as input for the second one.
    CvEM em_model2;
    params.cov_mat_type = CvEM::COV_MAT_DIAGONAL;
    params.start_step = CvEM::START_E_STEP;
    params.means = em_model.get_means();
    params.covs = em_model.get_covs();
    params.weights = em_model.get_weights();

    em_model2.train( samples, Mat(), params, &labels );
    // to use em_model2, replace em_model.predict()
    // with em_model2.predict() below
#endif







    // classify every image pixel
    for( i = 0; i < img.rows; i++ )
    {
        for( j = 0; j < img.cols; j++ )
        {
            sample.at<float>(0) = (float)j;
            sample.at<float>(1) = (float)i;
            cv::Mat probs;
            int response = cvRound(em_model.predict( sample , &probs ));//这个版本的得不到对数概率（这个值不仅考虑每个高斯分布下的概率，还结合每个高斯的权重得到一个总的概率，去对数后）
            Scalar c = colors[response];
            //std::cout<<"probs:"<<probs<<std::endl;
            //std::cout<<"response:"<<response<<std::endl;

//probs:[0.9531773613575238, 4.139578186930476e-14, 2.173447544698253e-19, 0.04682263864243473]
//            response:0





            circle( img, Point(j, i), 1, c*0.75, CV_FILLED );
        }
    }

    //draw the clustered samples
    for( i = 0; i < nsamples; i++ )
    {
        Point pt(cvRound(samples.at<float>(i, 0)), cvRound(samples.at<float>(i, 1)));
        circle( img, pt, 1, colors[labels.at<int>(i)], CV_FILLED );
    }

    imshow( "EM-clustering result", img );
    waitKey(0);

    return 0;
}
