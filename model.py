import torch
from torch import nn
import torch.nn.functional as F


    
class LECEN_32p(nn.Module):
        def __init__(self, num_pilots, Signal_length, Lidar_Length,squzeenum,user,B_H_size,H_size,over):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),  # 16 64
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),  # 16 64
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),  # 16 64
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),  # 16 64
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.output_layer = nn.Linear(4224, 512)
            self.output_layerop2 = nn.Linear(1024, 512)
            self.dropout1 = nn.Dropout(0.1) 
            self.bn1 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.relu = nn.ReLU()
            self.output_layer2 = nn.Linear(512, Signal_length)
            self.dropout2 = nn.Dropout(0.2)
            self.bn2 = nn.BatchNorm1d(Signal_length, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.relu = nn.ReLU()
            self.output_layer3 = nn.Linear(Signal_length+Lidar_Length, 256)
            self.bn3 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.output_layer4 = nn.Linear(256, H_size)
            #lidar
            self.LiDARExtract = LiDAR_Feature(Lidar_Length)
            # bottleneck
            self.bt1 = nn.Linear(Signal_length+Lidar_Length, int(squzeenum))
            self.bt2 = nn.Linear(int(squzeenum) ,int(squzeenum))
            self.bt3 = nn.Linear(int(squzeenum) ,Signal_length+Lidar_Length)
            self.sig = nn.Sigmoid()
            self.num_pilots = num_pilots
            self.over = over

        def forward(self, x, y): 
            #signal
            x = torch.reshape(x,[x.shape[0],2,32*self.over,self.num_pilots])
            x = x[:,:,:,0:self.num_pilots]
            x = x.transpose(1,3)
            x = self.layer3(x)
            x = x.reshape(x.size(0), -1)
            x = self.output_layer(x) 
            x = self.relu(x)
            x = self.bn1(x)
            x = self.output_layer2(x)         
            x = self.relu(x)   
            X = self.bn2(x)    
            #lidar
            l = self.LiDARExtract(y) 

            
            Multi_Feature = torch.cat((X, l),1)  
            u = self.bt1(Multi_Feature)
            u = self.relu(u)
            u = self.bt2(u)
            u = self.relu(u)
            u = self.bt3(u)
            u = self.sig(u)
            Multi_Feature2 = torch.mul(Multi_Feature,u)  
            
            output = self.output_layer3(Multi_Feature2)  
            output = self.relu(output)
            output = self.bn3(output)

            finaloutput = self.output_layer4(output)    

            return finaloutput
       
class LECEN_16p(nn.Module):
        def __init__(self, num_pilots, Signal_length, Lidar_Length,squzeenum,user,B_H_size,H_size,over):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),  # 16 64
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),  # 16 64
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),  # 16 64
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),  # 16 64
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.output_layer = nn.Linear(2112, 512)
            self.output_layerop2 = nn.Linear(1024, 512)
            self.dropout1 = nn.Dropout(0.1)  #!!!!!!!!!!!!!
            self.bn1 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.relu = nn.ReLU()
            self.output_layer2 = nn.Linear(512, Signal_length)
            self.dropout2 = nn.Dropout(0.2)
            self.bn2 = nn.BatchNorm1d(Signal_length, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.relu = nn.ReLU()
            self.output_layer3 = nn.Linear(Signal_length+Lidar_Length, 256)
            self.bn3 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.output_layer4 = nn.Linear(256, H_size)
            self.LiDARExtract = LiDAR_Feature(Lidar_Length)
            self.bt1 = nn.Linear(Signal_length+Lidar_Length, int(squzeenum))
            self.bt2 = nn.Linear(int(squzeenum) ,int(squzeenum))
            self.bt3 = nn.Linear(int(squzeenum) ,Signal_length+Lidar_Length)
            self.sig = nn.Sigmoid()
            self.num_pilots = num_pilots
            self.over = over

        def forward(self, x, y): 
            #signal
            x = torch.reshape(x,[x.shape[0],2,32*self.over,self.num_pilots])
            x = x[:,:,:,0:self.num_pilots]
            x = x.transpose(1,3)
            x = self.layer2(x)
            x = x.reshape(x.size(0), -1)
            x = self.output_layer(x)  
            x = self.relu(x)
            x = self.bn1(x)
            x = self.output_layer2(x)         
            x = self.relu(x)   
            X = self.bn2(x)   
            l = self.LiDARExtract(y)  
            
            Multi_Feature = torch.cat((X, l),1)  
            
            u = self.bt1(Multi_Feature)
            u = self.relu(u)
            u = self.bt2(u)
            u = self.relu(u)
            u = self.bt3(u)
            u = self.sig(u)
            
            weight_signal = torch.mean(u[:,0:512])
            weight_lidar = torch.mean(u[:,512:640])
            Multi_Feature2 = torch.mul(Multi_Feature,u)   
            output = self.output_layer3(Multi_Feature2)   
            output = self.relu(output)
            output = self.bn3(output)
            finaloutput = self.output_layer4(output)    
            return finaloutput


class LECEN_8p(nn.Module):
        def __init__(self, num_pilots, Signal_length, Lidar_Length,squzeenum,user,B_H_size,H_size,over):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),  # 16 64
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),  # 16 64
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),  # 16 64
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),  # 16 64
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.output_layer = nn.Linear(1056, 512)
            self.output_layerop2 = nn.Linear(1024, 512)
            self.dropout1 = nn.Dropout(0.1)  #!!!!!!!!!!!!!
            self.bn1 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.relu = nn.ReLU()
            self.output_layer2 = nn.Linear(512, Signal_length)
            self.dropout2 = nn.Dropout(0.2)
            self.bn2 = nn.BatchNorm1d(Signal_length, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.relu = nn.ReLU()
            self.output_layer3 = nn.Linear(Signal_length+Lidar_Length, 256)
            self.bn3 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.output_layer4 = nn.Linear(256, H_size)
            self.LiDARExtract = LiDAR_Feature(Lidar_Length)
            self.bt1 = nn.Linear(Signal_length+Lidar_Length, int(squzeenum))
            self.bt2 = nn.Linear(int(squzeenum) ,int(squzeenum))
            self.bt3 = nn.Linear(int(squzeenum) ,Signal_length+Lidar_Length)
            self.sig = nn.Sigmoid()
            self.num_pilots = num_pilots
            self.over = over

        def forward(self, x, y):  
            x = torch.reshape(x,[x.shape[0],2,32*self.over,self.num_pilots])
            x = x[:,:,:,0:self.num_pilots]
            x = x.transpose(1,3)
            x = self.layer1(x)
            x = x.reshape(x.size(0), -1)
            x = self.output_layer(x)  
            x = self.relu(x)
            x = self.bn1(x)
            x = self.output_layer2(x)         
            x = self.relu(x)   
            X = self.bn2(x)    
            l = self.LiDARExtract(y) 
            
            Multi_Feature = torch.cat((X, l),1)  
            
            u = self.bt1(Multi_Feature)
            u = self.relu(u)
            u = self.bt2(u)
            u = self.relu(u)
            u = self.bt3(u)
            u = self.sig(u)
            
            Multi_Feature2 = torch.mul(Multi_Feature,u)
            output = self.output_layer3(Multi_Feature2)  
            output = self.relu(output)
            output = self.bn3(output)
            finaloutput = self.output_layer4(output)   
            return finaloutput

        

class LiDAR_Feature(nn.Module):
    def __init__(self, LiDAR_Length):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),  #16 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),  #3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2), #3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2), #3
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)
        )
        
        
        self.output_layer = nn.Linear(2048, 1024) 
        self.bn1 = nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout1 = nn.Dropout(0.1) 
        self.relu = nn.ReLU()
        self.output_layer2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout2 = nn.Dropout(0.5)  
        self.relu = nn.ReLU()
        self.output_layer3 = nn.Linear(512, LiDAR_Length)
        self.bn3 = nn.BatchNorm1d(LiDAR_Length, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # bottleneck
        self.bt1 = nn.Linear(256, int(LiDAR_Length))
        self.bt2 = nn.Linear(int(LiDAR_Length) ,int(LiDAR_Length/2))
        self.bt3 = nn.Linear(int(LiDAR_Length/2) ,256)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        x = self.output_layer(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.output_layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        output = self.output_layer3(x)
        output = self.bn3(output)

        return output
    
     