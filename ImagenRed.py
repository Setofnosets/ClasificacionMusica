import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

def to_FullyConnected( name, s_filer=" ", n_filer=" ", offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, opacity=0.8, caption=" " , zlabelposition='midway'):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {RightBandedBox={
        name=""" + name +""",
        caption=""" +caption + """,
        xlabel={{ """+ '"'+str(n_filer) +'", "dummy"'+ """ }},
        zlabel="""+ str(s_filer) +""",
        fill=\FcColor,
        bandfill=\FcReluColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# Capas: Entrada: 64
# Capa convolucional: 128
# Capa de pooling: 2
# Capa convolucional: 256
# Capa de pooling: 2
# Capa convolucional: 256
# Capa de pooling: 2
# Capa Densa: 512
# Capa Densa: 256
# Capa de salida: 12
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    to_input('input.jpg'),
    to_Conv("input", 64, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2, caption="INPUT" ),
    to_Pool("pool", offset="(0,0,0)", to="(input-east)", height=32, depth=32, width=1),
    to_Conv("conv1", 128, 128, offset="(1,0,0)", to="(pool-east)", height=32, depth=32, width=2, caption="CONV" ),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)", height=16, depth=16, width=1),
    to_Conv("conv2", 256, 256, offset="(1,0,0)", to="(pool1-east)", height=16, depth=16, width=2, caption="CONV" ),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=8, depth=8, width=1),
    to_Conv("conv3", 256, 256, offset="(1,0,0)", to="(pool2-east)", height=8, depth=8, width=2, caption="CONV" ),
    to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)", height=4, depth=4, width=1),
    to_FullyConnected("dense1", n_filer=512 ,offset="(3,0,0)", to="(pool3-east)", caption="DENSE"  ),
    to_FullyConnected("dense2", n_filer=256 ,offset="(3,0,0)", to="(dense1-east)", caption="DENSE"  ),
    to_SoftMax("soft1", 12 ,"(3,0,0)", "(dense2-east)", caption="SOFTMAX"  ),
    to_connection("pool", "conv1"),

    to_connection("pool1", "conv2"),

    to_connection("pool2", "conv3"),

    to_connection("pool3", "dense1"),
    to_connection("dense1", "dense2"),
    to_connection("dense2", "soft1"),


    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()