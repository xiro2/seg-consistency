class DropPath(nn.Layer):
    
    def __init__(self, p=0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x

        Batch_size = x.shape[0]
        random_tensor = paddle.rand([Batch_size, 1, 1, 1])
        binary_mask = self.p < random_tensor

        x = x/(1 - self.p)
        x = x * binary_mask

        return x

class Block_base(nn.Layer):
    def __init__(self, in_channels):
        super(Block_base,self).__init__()
        out_channels=in_channels

        self.conv1=nn.Conv2D(in_channels=out_channels,out_channels=out_channels,kernel_size=3, stride=1, padding=1)
        self.norm1=nn.InstanceNorm2D(out_channels)
        self.relu1=nn.ReLU()

        self.conv2=nn.Conv2D(in_channels=out_channels,out_channels=out_channels,kernel_size=3, stride=1, padding=1)
        self.norm2=nn.InstanceNorm2D(out_channels)
        self.relu2=nn.ReLU()

 
    def forward(self, x):
        path = paddle.clone(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x+path)
        return x

class Block_down(nn.Layer):
    def __init__(self, in_channels):
        super(Block_down,self).__init__()
        self.conv1=nn.Conv2D(in_channels=in_channels,out_channels=in_channels*2,kernel_size=3, stride=2, padding=1)
        self.norm1=nn.InstanceNorm2D(in_channels*2)
        self.relu1=nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        return x       

class Block_up(nn.Layer):
    def __init__(self, in_channels):
        super(Block_up,self).__init__()
        self.up=nn.UpsamplingBilinear2D(scale_factor=2)
        self.conv1=nn.Conv2D(in_channels=in_channels,out_channels=int(in_channels/2),kernel_size=3, stride=1, padding=1)
        self.norm1=nn.InstanceNorm2D(int(in_channels/2))
        self.relu1=nn.ReLU()
    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        return x       

class Cyclegan(nn.Layer):
    def __init__(self,num_channels,in_channels=1,drop_prob=0):   
        super(Cyclegan,self).__init__()
        self.stem = nn.Conv2D(in_channels=in_channels,out_channels=num_channels,kernel_size=7,stride=1,padding=3,padding_mode='reflect')
        self.stemnorm = nn.InstanceNorm2D(num_channels)
        self.stemrelu = nn.ReLU()
        self.down1 = Block_down(in_channels=num_channels)
        self.down2 = Block_down(in_channels=num_channels*2)

        self.base1 = Block_base(in_channels=num_channels*4)
        self.base2 = Block_base(in_channels=num_channels*4)
        self.base3 = Block_base(in_channels=num_channels*4)
        self.base4 = Block_base(in_channels=num_channels*4)
        self.base5 = Block_base(in_channels=num_channels*4)
        self.base6 = Block_base(in_channels=num_channels*4)
        self.base7 = Block_base(in_channels=num_channels*4)
        self.base8 = Block_base(in_channels=num_channels*4)
        self.base9 = Block_base(in_channels=num_channels*4)

        self.up1 = Block_up(in_channels=num_channels*4)
        self.up2 = Block_up(in_channels=num_channels*2) 
        self.convout = nn.Conv2D(in_channels=num_channels*1,out_channels=1,kernel_size=7,stride=1,padding=3,padding_mode='reflect')
        self.act = nn.Tanh()
    def forward(self,inputs,mode='identity'):
        x = self.stem(inputs)
        x = self.stemnorm(x)
        x = self.stemrelu(x)

        x = self.down1(x)
        x = self.down2(x)
        
        x = self.base1(x)
        x = self.base2(x)
        x = self.base3(x)
        x = self.base4(x)
        x = self.base5(x)
        x = self.base6(x)
        x = self.base7(x)
        x = self.base8(x)
        x = self.base9(x)

        x = self.up1(x)         
        x = self.up2(x)
        x = self.convout(x)
        x = self.act(x)

        return x
       

class Block_discrim(nn.Layer):
    def __init__(self, in_channels,out_channels,stride,padding=1):
        super(Block_discrim,self).__init__()
        self.conv1=nn.Conv2D(in_channels=in_channels,out_channels=out_channels,kernel_size=4, stride=stride, padding=padding)
        self.norm1=nn.InstanceNorm2D(out_channels)
        self.relu1=nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        return x 

class Discriminator(nn.Layer):
    def __init__(self, in_channels=1,drop_prob=0,num_channels=64):
        super(Discriminator,self).__init__()
        self.block1 = Block_discrim(in_channels=1,out_channels=num_channels,stride=2)
        self.block2 = Block_discrim(in_channels=num_channels,out_channels=num_channels*2,stride=2)
        self.block3 = Block_discrim(in_channels=num_channels*2,out_channels=num_channels*4,stride=2)
        self.block4 = Block_discrim(in_channels=num_channels*4,out_channels=num_channels*8,stride=1)
        self.block5 = nn.Conv2D(in_channels=num_channels*8,out_channels=1,kernel_size=4, stride=1, padding=1)
    def forward(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = paddle.squeeze(x)

        return x

class MLP(nn.Layer):
    def __init__(self, in_features,out_features):
        super(MLP,self).__init__()
        self.dense1 = nn.Linear(in_features=in_features,out_features=out_features)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features=out_features,out_features=out_features)
    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.relu1(x)
        x = self.dense2(x)

        return x

class Encoder(nn.Layer):
    def __init__(self, num_channels, num_filters):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2D(in_channels=num_channels,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn1   = nn.BatchNorm2D(num_filters)
        
        self.conv2 = nn.Conv2D(in_channels=num_filters,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn2   = nn.BatchNorm2D(num_filters)

        self.pool  = nn.Conv2D(in_channels=num_filters,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.relu = nn.LeakyReLU()
    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x_conv = x           
        x_pool = self.pool(x)
        return x_conv, x_pool
    
    
class Decoder(nn.Layer):
    def __init__(self, num_channels, num_filters,norm):
        super(Decoder,self).__init__()
        self.up1 = nn.Conv2D(in_channels=num_channels,
                                    out_channels=num_filters,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.up2 = nn.UpsamplingBilinear2D(scale_factor=2)
        if(norm=='Instance'):
            self.bn1   = nn.BatchNorm2D(num_filters)
            self.bn2   = nn.BatchNorm2D(num_filters)
        else:
            self.bn1   = nn.BatchNorm2D(num_filters)
            self.bn2   = nn.BatchNorm2D(num_filters)
        self.conv1 = nn.Conv2D(in_channels=num_filters*2,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        
        self.conv2 = nn.Conv2D(in_channels=num_filters,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        
        self.relu = nn.LeakyReLU()
    def forward(self,input_conv,input_pool):
        x = self.up1(input_pool)
        x = self.up2(x)

        x = paddle.concat(x=[input_conv,x],axis=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class UNet(nn.Layer):
    def __init__(self,num_classes=5,num_channels=32):
        super(UNet,self).__init__()
        self.down1 = Encoder(num_channels=  1, num_filters=num_channels) 
        self.down2 = Encoder(num_channels=num_channels, num_filters=num_channels*2)
        self.down3 = Encoder(num_channels=num_channels*2, num_filters=num_channels*4)
        self.down4 = Encoder(num_channels=num_channels*4, num_filters=num_channels*8)
        
        self.mid_conv1 = nn.Conv2D(num_channels*8,num_channels*16,1)                 
        self.mid_bn1   = nn.BatchNorm2D(num_channels*16)
        self.relu1 = nn.LeakyReLU()
        self.mid_conv2 = nn.Conv2D(num_channels*16,num_channels*16,1)
        self.mid_bn2   = nn.BatchNorm2D(num_channels*16)
        self.relu2 = nn.LeakyReLU()
        self.up4 = Decoder(num_channels*16,num_channels*8,'Batch')                      
        self.up3 = Decoder(num_channels*8,num_channels*4,'Batch')
        self.up2 = Decoder(num_channels*4,num_channels*2,'Batch')
        self.up1 = Decoder(num_channels*2,num_channels,'Batch')
        self.last_conv = nn.Conv2D(num_channels,num_classes,kernel_size=1)        
        self.drop = nn.Dropout2D(p=0.9)
    def forward(self,inputs,mode='pure',drop='false'):
        if(mode=='pure'):
            x1, x = self.down1(inputs)
            x2, x = self.down2(x)
            x3, x = self.down3(x)
            x4, x = self.down4(x)
            x = self.mid_conv1(x)
            x = self.mid_bn1(x)
            x = self.relu1(x)
            x = self.mid_conv2(x)
            x = self.mid_bn2(x)
            x = self.relu2(x)
            if(drop=='true'):
                x = self.drop(x)
            x = self.up4(x4, x)
            x = self.up3(x3, x)
            x = self.up2(x2, x)
            x = self.up1(x1, x)
            x = self.last_conv(x)
            return x
        else:
            x1, x = self.down1(inputs)
            x2, x = self.down2(x)
            x3, x = self.down3(x)
            x4, x = self.down4(x)
            mlp3=paddle.clone(x)
            x = self.mid_conv1(x)
            x = self.mid_bn1(x)
            x = self.relu1(x)
            x = self.mid_conv2(x)
            x = self.mid_bn2(x)
            x = self.relu2(x)
            x = self.up4(x4, x)
            x = self.up3(x3, x)
            x = self.up2(x2, x)
            x = self.up1(x1, x)
    
        
            x = self.last_conv(x)
        
            return x,mlp3
