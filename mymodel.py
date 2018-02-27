from kaffe.tensorflow import Network

class GoogleNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .conv(1, 1, 64, 1, 1, name='reduction2')
             .conv(3, 3, 192, 1, 1, name='conv2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .max_pool(3, 3, 2, 2, name='pool2')
             .conv(1, 1, 96, 1, 1, name='icp1_reduction1')
             .conv(3, 3, 128, 1, 1, name='icp1_out1'))

        (self.feed('pool2')
             .conv(1, 1, 16, 1, 1, name='icp1_reduction2')
             .conv(5, 5, 32, 1, 1, name='icp1_out2'))

        (self.feed('pool2')
             .max_pool(3, 3, 1, 1, name='icp1_pool')
             .conv(1, 1, 32, 1, 1, name='icp1_out3'))

        (self.feed('pool2')
             .conv(1, 1, 64, 1, 1, name='icp1_out0'))

        (self.feed('icp1_out0', 
                   'icp1_out1', 
                   'icp1_out2', 
                   'icp1_out3')
             .concat(3, name='icp2_in')
             .conv(1, 1, 128, 1, 1, name='icp2_reduction1')
             .conv(3, 3, 192, 1, 1, name='icp2_out1'))

        (self.feed('icp2_in')
             .conv(1, 1, 32, 1, 1, name='icp2_reduction2')
             .conv(5, 5, 96, 1, 1, name='icp2_out2'))

        (self.feed('icp2_in')
             .max_pool(3, 3, 1, 1, name='icp2_pool')
             .conv(1, 1, 64, 1, 1, name='icp2_out3'))

        (self.feed('icp2_in')
             .conv(1, 1, 128, 1, 1, name='icp2_out0'))

        (self.feed('icp2_out0', 
                   'icp2_out1', 
                   'icp2_out2', 
                   'icp2_out3')
             .concat(3, name='icp2_out')
             .max_pool(3, 3, 2, 2, name='icp3_in')
             .conv(1, 1, 112, 1, 1, name='icp3_reduction1')
             .conv(3, 3, 224, 1, 1, name='icp3_out1'))

        (self.feed('icp3_in')
             .conv(1, 1, 24, 1, 1, name='icp3_reduction2')
             .conv(5, 5, 64, 1, 1, name='icp3_out2'))

        (self.feed('icp3_in')
             .max_pool(3, 3, 1, 1, name='icp3_pool')
             .conv(1, 1, 64, 1, 1, name='icp3_out3'))

        (self.feed('icp3_in')
             .conv(1, 1, 160, 1, 1, name='icp3_out0'))

        (self.feed('icp3_out0', 
                   'icp3_out1', 
                   'icp3_out2', 
                   'icp3_out3')
             .concat(3, name='icp3_out')
             .conv(1, 1, 160, 1, 1, name='icp4_reduction1')
             .conv(3, 3, 320, 1, 1, name='icp4_out1'))

        (self.feed('icp3_out')
             .conv(1, 1, 32, 1, 1, name='icp4_reduction2')
             .conv(5, 5, 128, 1, 1, name='icp4_out2'))

        (self.feed('icp3_out')
             .max_pool(3, 3, 1, 1, name='icp4_pool')
             .conv(1, 1, 128, 1, 1, name='icp4_out3'))

        (self.feed('icp3_out')
             .conv(1, 1, 256, 1, 1, name='icp4_out0'))

        (self.feed('icp4_out0', 
                   'icp4_out1', 
                   'icp4_out2', 
                   'icp4_out3')
             .concat(3, name='icp4_out')
             .avg_pool(5, 5, 3, 3, padding=None, name='cls3_pool')
             .conv(1, 1, 128, 1, 1, name='cls3_reduction')
             .fc(1024, name='cls3_fc1')
             .fc(7354, relu=False, name='cls3_fc2')
             .softmax(name='loss'))