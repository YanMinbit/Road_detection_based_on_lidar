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


#include "util_ppm.h"
#include <cstdio>
#include <cstring>

void writePGM ( const char* filename, int W, int H, const char* data )
{
	FILE* fp = fopen ( filename, "wb" );
	if ( !fp )
	{
		printf ( "Failed to open file '%s'!\n", filename );
		return;
	}
	fprintf ( fp, "P5\n%d %d\n%d\n", W, H, 255 );
	fwrite ( data, 1, W*H, fp );//用法https://www.cnblogs.com/xudong-bupt/p/3478297.html
	//size_t fwrite ( const void * ptr, size_t size, size_t count, FILE * stream );

//其中，ptr：指向保存数据的指针；size：每个数据类型的大小；count：数据的个数；stream：文件指针

//函数返回写入数据的个数。怎么感觉就是二进制流啊
	fclose ( fp );
}
unsigned char* readPPM ( const char* filename, int& W, int& H )
{
	FILE* fp = fopen ( filename, "rb" );
	if ( !fp )
	{
		printf ( "Failed to open file '%s'!\n", filename );
	}
	char hdr[256]={};
	size_t l=0;
	// Read the header
	char p,n;
	int D;
	while ( sscanf ( hdr, "%c%c %d %d %d", &p, &n, &W, &H, &D ) < 5 )// sscanf()会将参数str的字符串根据参数format字符串来转换并格式化数据。格式转换形式请参考scanf()。转换后的结果存于对应的参数内。

		//返回值 成功则返回参数数目，失败则返回-1，错误原因存于errno中。 返回0表示失败    否则，表示正确格式化数据的个数    例如：sscanf(str，"%d%d%s", &i,&i2, &s);    如果三个变成都读入成功会返回3。    如果只读入了第一个整数到i则会返回1。证明无法从str读入第二个整数。
	{
		fgets ( hdr+l, 256-l, fp );//为了避免缓冲区溢出，从终端读取输入时应当用fgets()代替gets()函数。fgets (buf, MAX, fp)
		char * comment = strchr ( hdr, 'p' );//原型为extern char *strchr(const char *s,char c)，可以查找字符串s中首次出现字符c的位置。
		if ( comment ) l = hdr - comment;//没明白这里为什么要进行这么复杂的处理，好像完全没必要吧？？？
		else l = strlen ( hdr );
		if ( l>=255 )
		{
			W=H=0;
			fclose ( fp );
			return NULL;
		}
	}//这个循环在第一次执行时因为数组里面没有数据所以返回的的是0，sscanf("1 2","%d %d %d",buf1, buf2, buf3); 成功调用返回值为2，即只有buf1，buf2成功转换。第二次里面有数据了，所以第二次没有执行就退出了
	if ( p != 'P' )
	{
		W=H=0;
		fclose ( fp );
		return NULL;
	}
	unsigned char * r = new unsigned char[W*H*3];
	if ( n=='6' )
		fread ( r, 1, W*H*3, fp );
	else if ( n=='3' )
	{
		int c;
		for ( int i=0; i<W*H*3; i++ )
		{
			fscanf ( fp, "%d", &c );
			r[i] = 255*c / D;
		}
	}
	else
	{
		W=H=0;
		fclose ( fp );
		return NULL;
	}
	fclose ( fp );//应该对ppm文件进行解码，需要是P、3／6，宽和高，D
	return r;
}
void writePPM ( const char* filename, int W, int H, unsigned char* data )
{
	FILE* fp = fopen ( filename, "wb" );//以二进制的方式打开文件
	if ( !fp )
	{
		printf ( "Failed to open file '%s'!\n", filename );
	}
	fprintf ( fp, "P6\n%d %d\n%d\n", W, H, 255 );//这里确实是先写了一个这样的头
	fwrite ( data, 1, W*H*3, fp );
	fclose ( fp );
}
