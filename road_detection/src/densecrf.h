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

#pragma once

#include <vector>
#include <cstdlib>

class PairwisePotential{
public:
	virtual ~PairwisePotential();//C++ 在继承中虚函数、纯虚函数、普通函数，三者的区别https://www.cnblogs.com/xudong-bupt/p/3570304.html
	virtual void apply( float * out_values, const float * in_values, float * tmp, int value_size ) const = 0;//纯虚函数
};
class SemiMetricFunction{
public:
	virtual ~SemiMetricFunction();
	// For two probabilities apply the semi metric transform: v_i = sum_j mu_ij u_j
	virtual void apply( float * out_values, const float * in_values, int value_size ) const = 0;
};


class DenseCRF{
protected:
	friend class BipartiteDenseCRF;//关于友元的相关知识https://www.cnblogs.com/zhuguanhao/p/6286145.html  这种友元类有必要吗，在一个类里面好友另一个类的对象不是什么问题吧？？？？
	//还是有必要的，否则私有的变量的调用会有问题
	// Number of variables and labels
	int N_, M_;//像素数和标签数
	float *unary_, *additional_unary_, *current_, *next_, *tmp_;//迭代过程中用到的能量和概率等
	
	// Store all pairwise potentials
	std::vector<PairwisePotential*> pairwise_;//有多少个特征（高斯核），这个向量就有多长
	
	// Run inference and return the pointer to the result
	float* runInference( int n_iterations, float relax);
	
	// Auxillary functions
	void expAndNormalize( float* out, const float* in, float scale = 1.0, float relax = 1.0 );
	
	// Don't copy this object, bad stuff will happen
	DenseCRF( DenseCRF & o ){}
public:
	// Create a dense CRF model of size N with M labels
	DenseCRF( int N, int M );
	virtual ~DenseCRF();
	// Add  a pairwise potential defined over some feature space
	// The potential will have the form:    w*exp(-0.5*|f_i - f_j|^2)
	// The kernel shape should be captured by transforming the
	// features before passing them into this function
	void addPairwiseEnergy( const float * features, int D, float w=1.0f, const SemiMetricFunction * function=NULL );
	
	// Add your own favorite pairwise potential (ownwership will be transfered to this class)
	void addPairwiseEnergy( PairwisePotential* potential );
	
	// Set the unary potential for all variables and labels (memory order is [x0l0 x0l1 x0l2 .. x1l0 x1l1 ...])
	void setUnaryEnergy( const float * unary );
	
	// Set the unary potential for a specific variable
	void setUnaryEnergy( int n, const float * unary );
	
	// Run inference and return the probabilities
	void inference( int n_iterations, float* result, float relax=1.0 );
	
	// Run MAP inference and return the map for each pixel
	void map( int n_iterations, short int* result, float relax=1.0 );//很重要
	
	// Step by step inference
	void startInference();
	void stepInference( float relax = 1.0 );
	void currentMap( short * result );
	
public: /* Debugging functions */
	// Compute the unary energy of an assignment
	void unaryEnergy( const short * ass, float * result );
	
	// Compute the pairwise energy of an assignment (half of each pairwise potential is added to each of it's endpoints)
	void pairwiseEnergy( const short * ass, float * result, int term=-1 );
};

class DenseCRF2D:public DenseCRF{//示例采用的是这个类，，下面的函数都调用了，最后得到分类结果调用了crf.map(10, map);
protected:
	// Width, height of the 2d grid
	int W_, H_;
public:
	// Create a 2d dense CRF model of size W x H with M labels
	DenseCRF2D( int W, int H, int M );//重要
	virtual ~DenseCRF2D();
	// Add a Gaussian pairwise potential with standard deviation sx and sy
	void addPairwiseGaussian( float sx, float sy, float w, const SemiMetricFunction * function=NULL );//重要
	
	// Add a Bilateral pairwise potential with spacial standard deviations sx, sy and color standard deviations sr,sg,sb
	void addPairwiseBilateral( float sx, float sy, float sr, float sg, float sb, const unsigned char * im, float w, const SemiMetricFunction * function=NULL );//重要
	
	// Set the unary potential for a specific variable
	void setUnaryEnergy( int x, int y, const float * unary );//这个是带off-set项的
	using DenseCRF::setUnaryEnergy;//这个是直接将unary拷到类的成员变量里面
};

// A dense CRF in a bipartite graph
class BipartiteDenseCRF{//这个好像在cpp里面没有实现啊？？？有，在bipartitedensecrf.cpp里面
protected:
	// Two dense CRF's that are connected by a set of completely connected edges (in a bipartite graph)
	DenseCRF* dense_crfs_[2];//DenseCRF是本类的友元类
	
	// Number of variables and labels
	int N_[2], M_;
	
	// All bipartite pairwise potentials (all others are stored in each dense_crfs respectively)
	std::vector<PairwisePotential*> pairwise_[2];
	
	// Don't copy this object, bad stuff will happen
	BipartiteDenseCRF( BipartiteDenseCRF & o ){}
	
	// Run inference and return the pointer to the result
	void runInference( int n_iterations, float ** prob, float relax);
public:
	// Create a dense CRF model of size N with M labels
	BipartiteDenseCRF( int N1, int N2, int M );
	~BipartiteDenseCRF();
	
	// Add  a pairwise potential defined over some feature space
	// The potential will have the form:    w*exp(-0.5*|f_i - f_j|^2)
	// The kernel shape should be captured by transforming the
	// features before passing them into this function
	void addPairwiseEnergy( const float * features1, const float * features2, int D, float w=1.0f, const SemiMetricFunction * function=NULL );
	
	// Add your own favorite pairwise potential (ownwership will be transfered to this class)
	void addPairwiseEnergy( PairwisePotential* potential12, PairwisePotential* potential21 );
	
	// Run inference and return the probabilities
	void inference( int n_iterations, float* result1, float * result2, float relax=1 );
	
	// Run MAP inference and return the map for each pixel
	void map( int n_iterations, short int* result1, short int* result2, float relax=1 );
	
	// Access the two CRF's directly
	DenseCRF& getCRF( int i );
	const DenseCRF& getCRF( int i ) const;
	
	// Step by step inference
	void startInference();
	void stepInference( float relax = 1.0 );
	void currentMap( short * result );
};



// This function defines a simplified interface to the permutohedral lattice
// We assume a filter standard deviation of 1
class Permutohedral;
class Filter{
protected:
    int n1_, o1_, n2_, o2_;
    Permutohedral * permutohedral_;
    // Don't copy
    Filter( const Filter& filter ){}
public:
    // Use different source and target features
    Filter( const float * source_features, int N_source, const float * target_features, int N_target, int feature_dim );
    // Use the same source and target features
    Filter( const float * features, int N, int feature_dim );
    //
    ~Filter();
    // Filter a bunch of values
    void filter( const float * source, float * target, int value_size );
};
