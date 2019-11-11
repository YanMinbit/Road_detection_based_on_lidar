//这个文件是我自己添加的  
//minima@minima-ThinkPad-T450:~/cppdoc/densecrf$ grep -n -H -R "__SSE__"
//上面语句可以查找在当前文件夹下含有的字段的位置和文件名


#pragma once

//#define __SSE__  //400多毫秒 默认是有定义的，所以这一句不需要
//#undef __SSE__   //如果不加这句默认是有定义的，也就是使用了SSE加速，加了这句，那么才等于没有定义，这个时候的运行时间是500多毫秒
