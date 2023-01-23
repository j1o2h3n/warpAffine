import * as tf from '@tensorflow/tfjs';
import * as math from "mathjs";


function warpAffine(src, matrix, dsize=[112,112],borderValue=[0,0,0],bilinear_interpolation=true) // 仿射变换
{
    /*
    :param src: tensor, shape[H, W, C]
    :param matrix: Tuple, 仿射矩阵. shape[2, 3]
    :param dsize: Tuple,shape[W, H]. 输出的size
    :param borderValue: Tuple,空白处填充值,[0,0,0] or [255,255,255] etc.
    :param bilinear_interpolation: bool,是否选择双线性插值
    :return: tensor, shape[dsize[1], dsize[0], C]
    References：
    https://blog.csdn.net/weixin_42398658/article/details/121019668
    https://blog.csdn.net/qq_40939814/article/details/117966835
    */

    let row=[],col=[]

    matrix=matrix.concat([[0,0,1]])
    matrix=math.inv(matrix)
    matrix.pop()
    matrix = matrix //矩阵求逆

    for(let i=0; i<dsize[0]; i++){row.push(i)} //可以用tf.range
    for(let i=0; i<dsize[0]; i++){col.push(i)}
    const [grid_x, grid_y] = tf.meshgrid(row, col) //生成目标xy坐标

    //将目标坐标映射源坐标上
    let _src_x = tf.add(tf.add(tf.mul(matrix[0][0], grid_x), tf.mul(matrix[0][1], grid_y)), matrix[0][2]) // X. dst -> src
    let _src_y = tf.add(tf.add(tf.mul(matrix[1][0], grid_x), tf.mul(matrix[1][1], grid_y)), matrix[1][2]) // Y. dst -> src

    dsize.push(3) //添加特征维度
    let output = tf.buffer(dsize) //创建输出的空张量,存放在缓冲区
    
    if(bilinear_interpolation) //进行双线性插值
    {
        //从张量转成数组存储
        _src_x = _src_x.arraySync()
        _src_y = _src_y.arraySync()
        src = src.arraySync()

        let low_src_x = math.floor(_src_x) //向下取整
        let high_src_x = math.ceil(_src_x)
        let low_src_y = math.floor(_src_y) 
        let high_src_y = math.ceil(_src_y) //向上取整

        let pos_x = math.subtract(_src_x, low_src_x) //获取相对位置
        let pos_y = math.subtract(_src_y, low_src_y)

        let p0_area // 预定义占比面积变量
        let p1_area
        let p2_area
        let p3_area
        
        let p_value // 预定义中心像素值变量

        for(let i=0; i<dsize[0]; i++) //y坐标
        {
            for(let j=0; j<dsize[1]; j++) //x坐标
            {
                if( (0<=low_src_x[i][j]) && (high_src_x[i][j]<src[0].length) && (0 <= low_src_y[i][j]) && (high_src_y[i][j] < src.length) ) //如果目标坐标映射后在源图范围内，将源图像素赋给目标位置，但是这里对边界值的想法还有点可以再优化一下，超出范围一点点的可以通过向邻近点进行取值，而不是舍去赋0
                {
                    // 双线性插值
                    // p0        p1
                    //       p
                    // p2        p3
                    p0_area=(1 - pos_x[i][j]) * (1 - pos_y[i][j])
                    p1_area=(    pos_x[i][j]) * (1 - pos_y[i][j])
                    p2_area=(1 - pos_x[i][j]) * (    pos_y[i][j])
                    p3_area=(    pos_x[i][j]) * (    pos_y[i][j])

                    for(let k=0; k<3; k++) //循环三次选择通道
                    {
                        p_value = ((src[low_src_y[i][j]][low_src_x[i][j]][k])*p0_area) + ((src[low_src_y[i][j]][high_src_x[i][j]][k])*p1_area)
                                + ((src[high_src_y[i][j]][low_src_x[i][j]][k])*p2_area) + ((src[high_src_y[i][j]][high_src_x[i][j]][k])*p3_area) //双线性插值
                        output.set(p_value,  i, j, k)
                    }
                }
                else
                {
                    output.set(borderValue[0],  i, j, 0)
                    output.set(borderValue[1],  i, j, 1)
                    output.set(borderValue[2],  i, j, 2)
                }
            }
        } 
    }
    else //不采用双线性插值方式，直接对其进行四舍五入取整值
    {
        let src_x = tf.round(_src_x) //取整
        let src_y = tf.round(_src_y)
        
        let src_x_clip = tf.clipByValue(src_x, 0, src.shape[1]-1) //截取,得到合法索引
        let src_y_clip = tf.clipByValue(src_y, 0, src.shape[0]-1)

        //从张量转成数组存储
        src_x_clip = src_x_clip.arraySync()
        src_y_clip = src_y_clip.arraySync()
        src_x = src_x.arraySync()
        src_y = src_y.arraySync()
        src = src.arraySync()

        for(let i=0; i<dsize[0]; i++) //y坐标
        {
            for(let j=0; j<dsize[1]; j++) //x坐标
            {
                if( (0<=src_x[i][j]) && (src_x[i][j]<src[0].length) && (0 <= src_y[i][j]) && (src_y[i][j] < src.length) ) //如果目标坐标映射后在源图范围内，将源图像素赋给目标位置
                {
                    // output.set(tf.slice(src,[src_y_clip[i][j],src_x_clip[i][j],0],[1,1,1]), src_y_clip[i][j], src_x_clip[i][j], 0)
                    // output.set(tf.slice(src,[src_y_clip[i][j],src_x_clip[i][j],1],[1,1,1]), src_y_clip[i][j], src_x_clip[i][j], 1)
                    // output.set(tf.slice(src,[src_y_clip[i][j],src_x_clip[i][j],2],[1,1,1]), src_y_clip[i][j], src_x_clip[i][j], 2)
                    output.set(src[src_y_clip[i][j]][src_x_clip[i][j]][0],  i, j, 0)
                    output.set(src[src_y_clip[i][j]][src_x_clip[i][j]][1],  i, j, 1)
                    output.set(src[src_y_clip[i][j]][src_x_clip[i][j]][2],  i, j, 2)
                }
                else
                {
                    output.set(borderValue[0],  i, j, 0)
                    output.set(borderValue[1],  i, j, 1)
                    output.set(borderValue[2],  i, j, 2)
                }
            }
        } 
    }      

    output = output.toTensor() //将缓冲区转换回张量

    output = tf.round(output) //对像素值取整
    output = tf.clipByValue(output, 0, 255) //截取像素值范围

    return output
}