// This code here comes from http://www.xnoiz.co.cc/fast-exp-x/

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
#include "define.h"

inline float fast_log2 (float val) {//应该是2为底的对数函数有这样的快速算法，利用这个可以加快其他为底的对数计算，也就是再乘个常数而已
   int * const  exp_ptr = reinterpret_cast <int *> (&val);
   int          x = *exp_ptr;
   const int    log_2 = ((x >> 23) & 255) - 128;
   x &= ~(255 << 23);
   x += 127 << 23;
   *exp_ptr = x;

   return (val + log_2);
}
inline float fast_log (const float &val) {//注意代码里面的log表示的是自然对数
   return (fast_log2 (val) * 0.69314718f);//数学上：log2(a)=ln(a)/ln(2)，而ln(2)=0.69314718f
}
inline float very_fast_exp(float x) {//还有指数函数的快速计算方法，但是下面提到了这个函数只应对本解决方案，因为考虑了本应用的数据范围
	// err <= 3e-3
// 	return 1
// 		   -x*(0.9664
// 		   -x*(0.3536));

	return 1
	-x*(0.9999999995f
	-x*(0.4999999206f

	-x*(0.1666653019f
	-x*(0.0416573475f
	-x*(0.0083013598f

	-x*(0.0013298820f
	-x*(0.0001413161f)))))));
}
inline float fast_exp(float x) {
	bool lessZero = true;
	if (x < 0) {

		lessZero = false;
		x = -x;

	}
	// This diry little trick only works because of the normalization and the fact that one element in the normalization is 1
	if (x > 20)
		return 0;
	int mult = 0;

	while (x > 0.69*2*2*2) {

		mult+=3;
		x /= 8.0f;
	}

	while (x > 0.69*2*2) {
		mult+=2;

		x /= 4.0f;
	}
	while (x > 0.69) {

		mult++;
		x /= 2.0f;
	}

	x = very_fast_exp(x);
	while (mult) {

		mult--;
		x = x*x;
	}

	if (lessZero) {
		return 1 / x;

	} else {
		return x;
	}
}
//SSE 能实现稍微的并行化，特点处于C 和汇编之间
//SSE是指令集的简称，它包括70条指令，其中包含单指令多数据浮点计算、以及额外的SIMD整数和高速缓存控制指令。其优势包括：更高分辨率的图像浏览和处理、高质量音频、MPEG2视频、同时MPEG2加解密；语音识别占用更少CPU资源；更高精度和更快响应速度。
//https://blog.csdn.net/mydear_11000/article/details/50856590
#ifdef __SSE__   //https://blog.csdn.net/taily_duan/article/details/53392060
#include <xmmintrin.h>
#include "sse_defs.h"
inline __m128 very_fast_exp(__m128 x) {
	return _mm_set1_ps(1)
	-x*(_mm_set1_ps(0.9999999995)
	-x*(_mm_set1_ps(0.4999999206)

	-x*(_mm_set1_ps(0.1666653019)
	-x*(_mm_set1_ps(0.0416573475)
	-x*(_mm_set1_ps(0.0083013598)

	-x*(_mm_set1_ps(0.0013298820)
	-x*(_mm_set1_ps(0.0001413161))))))));
}
inline __m128 fast_exp(__m128 x) {
	const __m128 zero = _mm_set1_ps( 0.0f );
	__m128 lessZero = _mm_cmplt_ps( zero, x );
	__m128 one      = _mm_set1_ps(1.0);
	// Invert x of less than 0
	x = _mm_add_ps( _mm_and_ps(lessZero,x), _mm_andnot_ps(lessZero,-x) );
	
	// This diry little trick only works because of the normalization and then fact that one element in the normalization is 1
	__m128 set_zero = _mm_cmple_ps( x, _mm_set1_ps(20.0) );
	
	// Bring it into range
	__m128 mult = zero;
	for( int i=4, j=2; i>0; i/=2, j-- ){
		__m128 cmp = _mm_cmpgt_ps( x, _mm_set1_ps( 0.69*i ) );
		mult += _mm_and_ps( cmp, _mm_set1_ps( j+1 ) );
		x = x / (_mm_and_ps( cmp, _mm_set1_ps( 2*i-1 )) + one);
	}
	
	x = very_fast_exp(x);
	
	while(1){
		__m128 cmp = _mm_cmpgt_ps( mult, _mm_set1_ps(0.0) );
		if (!_mm_movemask_ps(cmp)) break;
		mult -= _mm_and_ps( cmp, one );
		x *= (_mm_and_ps( cmp, x-one ) + one);
	}
	
	x = _mm_add_ps( _mm_and_ps(lessZero,one/x), _mm_andnot_ps(lessZero,x) );
	return _mm_and_ps( set_zero, x );
}
#endif
