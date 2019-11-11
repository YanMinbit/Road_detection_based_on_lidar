/*
    Copyright (c) 2011, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "densecrf.h"
#include "fastmath.h"
#include "permutohedral.h"
#include "util.h"
#include <cmath>
#include <cstring>
#include <iostream>

PairwisePotential::~PairwisePotential() {
}
SemiMetricFunction::~SemiMetricFunction() {
}

//外部调用：构造特征的时候会调用构造函数一次，然后每次推理的时候会调用apply一次(为了方便可以先假设只有一个特征函数)
class PottsPotential: public PairwisePotential{//重点就是一个构造函数和一个apply函数，每个特征会对应有一个这样的对象
protected:
	Permutohedral lattice_;//这个成员变量是关键，传进去的参数包括 特征向量数组，特征维度，像素数，当前像素标签概率数组和标签数
	PottsPotential( const PottsPotential&o ){}
	int N_;//像素数
	float w_;//权值
	float *norm_;//承接Permutohedral处理的结果
public:
	~PottsPotential(){
		deallocate( norm_ );
	}
	PottsPotential(const float* features, int D, int N, float w, bool per_pixel_normalization=true) :N_(N), w_(w) {//得到成对的连接energy
		//per_pixel_normalization 表示高斯核或者特征部分的归一化，像素归一化是指针对特征函数每个和同一个像素连接的高斯量求和，而全局归一化是指对所有像素和所有连接高斯量求和然后除以像素数
		lattice_.init( features, D, N );//这个就是参考文献的代码了 features的大小是W*H*特征维度 D表示特征维度，如外观部分是5维的，光滑部分是2维的
		norm_ = allocate( N );//这个量应该是某种中间量，算一次后面没有变化了，所以这个变量到底是指什么在应用扩展的时候还是需要搞清楚的
		for ( int i=0; i<N; i++ )
			norm_[i] = 1;
		// Compute the normalization factor
		lattice_.compute( norm_, norm_, 1 );//注意这里的维度和下面apply的维度是不一样的，根据apply的调用，中间一个量应该是概率意义上的量
		if ( per_pixel_normalization ) {
			// use a per pixel normalization
			for ( int i=0; i<N; i++ )
				norm_[i] = 1.f / (norm_[i]+1e-20f);//norm_跟高斯核相关的一个变量，应该是每个像素对应和别的像素连接形成的特征量的和，但是通过这一步后应该变成了倒数
		}
		else {//这样操作应该是取的文章里面归一化量的倒数
			float mean_norm = 0;
			for ( int i=0; i<N; i++ )
				mean_norm += norm_[i];
			mean_norm = N / mean_norm;
			// use a per pixel normalization
			for ( int i=0; i<N; i++ )
				norm_[i] = mean_norm;//这个每个像素弄成一样效果肯定不会好啊
		}
	}
	void apply(float* out_values, const float* in_values, float* tmp, int value_size) const {//父类里面是纯虚函数，但是这里已经进行了定义
		lattice_.compute( tmp, in_values, value_size );//这个应该执行的是Message passing 这一步吧  所以最难的就是这个了
		//value_size是标签数。in_values是概率，每次推理迭代的时候调用pairwise_[i]->apply( next_, current_, tmp_, M_ );用current_更新next_，next_还是能量，而current_是概率
		for ( int i=0,k=0; i<N_; i++ )//i表示像素位置  这里实现的是Compatibility transform和local Update(不包括求e的指数)
			for ( int j=0; j<value_size; j++, k++ )//j表示标签位置 K 表示整个数组中的索引
				out_values[k] += w_*norm_[i]*tmp[k];//注意这里的能量是加了负号的！！！！ 权值是从外部调用传过来的，也就是这里的实现没有参数学习的过程 对一个特征函数，每个像素每个标签这里都会计算一次，而多个特征这个函数会对应的调用多次，从文章中看，一个是加权参数，一个应该是特征遍历连接得到的Q（可能是一个是Q一个是高斯对应的部分），所以tmp是什么？？是表示标签一致性的吗？？
	}
};
class SemiMetricPotential: public PottsPotential{//SemiMetricPotential和上面的PottsPotential有什么区别？？？可以肯定的是是一种高斯分布的能量关系（对应一种特征）
protected:
	const SemiMetricFunction * function_;
public:
	void apply(float* out_values, const float* in_values, float* tmp, int value_size) const {//这个继承了还能再定义啊？？？？？
		//c++规定，当一个成员函数被声明为虚函数后，其派生类的同名函数都自动成为虚函数。因此在派生类重新声明该

		//虚函数时，可以加virtual，也可以不加，但习惯上一般在每层声明该函数时都加上virtual，使程序更加清晰。
		lattice_.compute( tmp, in_values, value_size );

		// To the metric transform
		float * tmp2 = new float[value_size];
		for ( int i=0; i<N_; i++ ) {
			float * out = out_values + i*value_size;
			float * t1  = tmp  + i*value_size;
			function_->apply( tmp2, t1, value_size );
			for ( int j=0; j<value_size; j++ )
				out[j] -= w_*norm_[i]*tmp2[j];
		}
		delete[] tmp2;
	}
	SemiMetricPotential(const float* features, int D, int N, float w, const SemiMetricFunction* function, bool per_pixel_normalization=true) :PottsPotential( features, D, N, w, per_pixel_normalization ),function_(function) {
	}//注意构造函数上做了这样的处理
};



/////////////////////////////
/////  Alloc / Dealloc  /////
/////////////////////////////
DenseCRF::DenseCRF(int N, int M) : N_(N), M_(M) {//N表示总的像素个数
	unary_ = allocate( N_*M_ );
	additional_unary_ = allocate( N_*M_ );
	current_ = allocate( N_*M_ );
	next_ = allocate( N_*M_ );
	tmp_ = allocate( 2*N_*M_ );
	// Set the additional_unary_ to zero
	memset( additional_unary_, 0, sizeof(float)*N_*M_ );
}
DenseCRF::~DenseCRF() {
	deallocate( unary_ );
	deallocate( additional_unary_ );
	deallocate( current_ );
	deallocate( next_ );
	deallocate( tmp_ );
	for( unsigned int i=0; i<pairwise_.size(); i++ )
		delete pairwise_[i];
}
DenseCRF2D::DenseCRF2D(int W, int H, int M) : DenseCRF(W*H,M), W_(W), H_(H) {//根据给定数据开辟一些数组变量
}
DenseCRF2D::~DenseCRF2D() {
}




/////////////////////////////////
/////  Pairwise Potentials  /////
/////////////////////////////////
void DenseCRF::addPairwiseEnergy (const float* features, int D, float w, const SemiMetricFunction * function) {
	if (function)
		addPairwiseEnergy( new SemiMetricPotential( features, D, N_, w, function ) );
	else
		addPairwiseEnergy( new PottsPotential( features, D, N_, w ) );
}
void DenseCRF::addPairwiseEnergy ( PairwisePotential* potential ){//PairwisePotential这个类里面有纯虚函数，所以不能new,但是可以转化子类变量
	pairwise_.push_back( potential );
}
void DenseCRF2D::addPairwiseGaussian ( float sx, float sy, float w, const SemiMetricFunction * function ) {//最后这个参数外部调用的时候并没有传递这个参数，之所以可以这样是因为头文件里面的声明有给定默认值
	float * feature = new float [N_*2];//N_表示像素数
	for( int j=0; j<H_; j++ )
		for( int i=0; i<W_; i++ ){
			feature[(j*W_+i)*2+0] = i / sx;//参考文献的开源代码就是这样的，现在还没有理解
			feature[(j*W_+i)*2+1] = j / sy;
		}
	addPairwiseEnergy( feature, 2, w, function );//这里应该在有没有传递最后一个参数的情况下调用的操作是不同的吧
	delete [] feature;
}
//下面这个函数和上面的函数算法上没有看到区别啊，只是考虑的特征向量不同了，上面是纯位置，下面是位置和颜色
void DenseCRF2D::addPairwiseBilateral ( float sx, float sy, float sr, float sg, float sb, const unsigned char* im, float w, const SemiMetricFunction * function ) {
	float * feature = new float [N_*5];
	for( int j=0; j<H_; j++ )
		for( int i=0; i<W_; i++ ){
			feature[(j*W_+i)*5+0] = i / sx;
			feature[(j*W_+i)*5+1] = j / sy;
			feature[(j*W_+i)*5+2] = im[(i+j*W_)*3+0] / sr;
			feature[(j*W_+i)*5+3] = im[(i+j*W_)*3+1] / sg;
			feature[(j*W_+i)*5+4] = im[(i+j*W_)*3+2] / sb;
		}
	addPairwiseEnergy( feature, 5, w, function );
	delete [] feature;
}




//////////////////////////////
/////  Unary Potentials  /////
//////////////////////////////
void DenseCRF::setUnaryEnergy ( const float* unary ) {
	memcpy( unary_, unary, N_*M_*sizeof(float) );
}
void DenseCRF::setUnaryEnergy ( int n, const float* unary ) {
	memcpy( unary_+n*M_, unary, M_*sizeof(float) );
}
void DenseCRF2D::setUnaryEnergy ( int x, int y, const float* unary ) {
	memcpy( unary_+(x+y*W_)*M_, unary, M_*sizeof(float) );
}



///////////////////////
/////  Inference  /////
///////////////////////
void DenseCRF::inference ( int n_iterations, float* result, float relax ) {
	// Run inference
	float * prob = runInference( n_iterations, relax );
	// Copy the result over
	for( int i=0; i<N_; i++ )
		memcpy( result+i*M_, prob+i*M_, M_*sizeof(float) );
}
void DenseCRF::map ( int n_iterations, short* result, float relax ) {//头文件里面的声明是这样的：void map( int n_iterations, short int* result, float relax=1.0 );
	// Run inference
	float * prob = runInference( n_iterations, relax );
	
	// Find the map
	for( int i=0; i<N_; i++ ){
		const float * p = prob + i*M_;//每个像素对应M个标签，每个标签都有一个概率
		// Find the max and subtract it so that the exp doesn't explode
		float mx = p[0];//记录每个像素的最大概率值
		int imx = 0;//最大给率对应的标签索引
		for( int j=1; j<M_; j++ )
			if( mx < p[j] ){
				mx = p[j];
				imx = j;
			}
		result[i] = imx;//所以得到的结果是每个像素取的最可能的标签索引
	}
}
float* DenseCRF::runInference( int n_iterations, float relax ) {//relax默认是1，就是完全的用新估计代替历史的估计，没有记忆
	startInference();//准备开始推理 ，其实就做了一件事情：用开始的能量初始化当前的概率
	for( int it=0; it<n_iterations; it++ )
	{
		stepInference(relax);//单步执行推理
		//std::cout<<it<<std::endl;
	}
	return current_;//DenseCRF里面受保护的成员变量，应该记录的是当前迭代步骤对应的每个像素属于每个标签的概率，对应文章中应该是Q，与能量函数是相关的
}
void DenseCRF::expAndNormalize ( float* out, const float* in, float scale, float relax ) {
	//推理初始时调用expAndNormalize( current_, unary_, -1 )，为什么这里传的是-1？？unary是正正常常的概率，是正的（错了，应该是概率对应的能量，E=-logP,P=e(-E)括号里是指数）;单步推理时调用expAndNormalize( current_, next_, 1.0, relax );
	//current_, next_两者都是每个像素每个标记都有一个概率值与之对应，所以这里是根据输入的Q做一些操作，但第一次准备推理的时候unary是能量
	float *V = new float[ N_+10 ];//这个不需要定义这么大吧？？
	for( int i=0; i<N_; i++ ){
		const float * b = in + i*M_;
		// Find the max and subtract it so that the exp doesn't explode
		float mx = scale*b[0];
		for( int j=1; j<M_; j++ )
			if( mx < scale*b[j] )
				mx = scale*b[j];
		float tt = 0;
		for( int j=0; j<M_; j++ ){
			V[j] = fast_exp( scale*b[j]-mx );//怎么推出来采用unary更新current的时候，unary越大的标签更新后current反而越小，因为unary是能量，本来就是能量越大概率越小
			tt += V[j];
		}
		// Make it a probability
		for( int j=0; j<M_; j++ )
			V[j] /= tt;//如果输入的N＊M的数据不满足概率为一的条件经过这一步就能转化为概率表示，其实就是归一化
		
		float * a = out + i*M_;
		for( int j=0; j<M_; j++ )
			if (relax == 1)
				a[j] = V[j];
			else
				a[j] = (1-relax)*a[j] + relax*V[j];//relax（默认是1）这个变量表示用输入概率更新输出概率时输出概率占比
	}
	delete[] V;
}
///////////////////
/////  Debug  /////
///////////////////

void DenseCRF::unaryEnergy(const short* ass, float* result) {
	for( int i=0; i<N_; i++ )
		if ( 0 <= ass[i] && ass[i] < M_ )
			result[i] = unary_[ M_*i + ass[i] ];
		else
			result[i] = 0;
}
void DenseCRF::pairwiseEnergy(const short* ass, float* result, int term) {
	float * current = allocate( N_*M_ );
	// Build the current belief [binary assignment]
	for( int i=0,k=0; i<N_; i++ )
		for( int j=0; j<M_; j++, k++ )
			current[k] = (ass[i] == j);
	
	for( int i=0; i<N_*M_; i++ )
		next_[i] = 0;
	if (term == -1)
		for( unsigned int i=0; i<pairwise_.size(); i++ )
			pairwise_[i]->apply( next_, current, tmp_, M_ );
	else
		pairwise_[ term ]->apply( next_, current, tmp_, M_ );
	for( int i=0; i<N_; i++ )
		if ( 0 <= ass[i] && ass[i] < M_ )
			result[i] =-next_[ i*M_ + ass[i] ];
		else
			result[i] = 0;
	deallocate( current );
}
void DenseCRF::startInference(){//推理之前进行初始化
	// Initialize using the unary energies
	expAndNormalize( current_, unary_, -1 );//其实就是用开始的能量初始化当前的概率
}
void DenseCRF::stepInference( float relax ){
#ifdef SSE_DENSE_CRF
	__m128 * sse_next_ = (__m128*)next_;
	__m128 * sse_unary_ = (__m128*)unary_;
	__m128 * sse_additional_unary_ = (__m128*)additional_unary_;
#endif
	// Set the unary potential
#ifdef SSE_DENSE_CRF
	for( int i=0; i<(N_*M_-1)/4+1; i++ )
		sse_next_[i] = - sse_unary_[i] - sse_additional_unary_[i];
#else
	for( int i=0; i<N_*M_; i++ )//搞清楚下面每次迭代执行会不会有变化，没有变化，这个在apply的时候会和current加到一起
		next_[i] = -unary_[i] - additional_unary_[i];//additional_unary_默认是0.这里倒是都加了一个负号  E=-logP,P=e(-E)括号里是指数
#endif
	
	// Add up all pairwise potentials  (为了方便可以先假设只有一个特征函数)
	for( unsigned int i=0; i<pairwise_.size(); i++ )//pairwise_的数量应该就是高斯核的数量也是特征函数的个数    这一步才是真真推理的地方，应该是包括文章里面的算法1和2
		pairwise_[i]->apply( next_, current_, tmp_, M_ );//我们调用的时候没有传入SemiMetricFunction这个参数，这里调用的是PottsPotential里面的函数
	
	// Exponentiate and normalize
	expAndNormalize( current_, next_, 1.0, relax );//local Update的求e的指数和归一化
	//这里传1和上面的-1倒是说的通，这里确定next_还是能量，而current_是概率吗？？是的
}
void DenseCRF::currentMap( short * result ){
	// Find the map
	for( int i=0; i<N_; i++ ){
		const float * p = current_ + i*M_;
		// Find the max and subtract it so that the exp doesn't explode
		float mx = p[0];
		int imx = 0;
		for( int j=1; j<M_; j++ )
			if( mx < p[j] ){
				mx = p[j];
				imx = j;
			}
		result[i] = imx;
	}
}
