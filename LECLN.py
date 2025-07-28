import torch
import numpy as np
import model
from torch import nn
from torch.utils.data import DataLoader  
import dataset_Multitemp
import scipy.io

def get_batch_NMSE2(Hhat, H, Antenna):
    H = H.cpu().numpy()
    H_real = H[:,:int(Antenna)]
    H_imag = H[:,int(Antenna):]
    H = H_real + 1j*H_imag
    up = np.sum(abs(Hhat - H)**2, axis=1)
    down = np.sum(abs(H)**2, axis=1)
    batch_result = 10*np.log10(up/down)
    print(batch_result)
    print('----------------------------------------------')
    return np.mean(batch_result)

def get_batch_MSE2(Hhat, H, Antenna):
    H = H.cpu().numpy()
    H_real = H[:,:int(Antenna)]
    H_imag = H[:,int(Antenna):]
    H = H_real + 1j*H_imag
    up = np.sum(abs(Hhat - H)**2, axis=1)
    down = np.sum(abs(H)**2, axis=1)
    batch_result = (up/down)
    print(batch_result)
    print('----------------------------------------------')
    return np.mean(batch_result)

def scale_three_channels(tensor):
    mean = tensor.mean()
    std = tensor.std()
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

def get_batch_NMSE_OFDM(Hhat, H, Antenna):
    H = H.cpu().numpy()
    H_real = H[:,:int(Antenna)]
    H_imag = H[:,int(Antenna):]
    H = H_real + 1j*H_imag
    up = np.sum(abs(Hhat - H)**2, axis=1)
    down = np.sum(abs(H)**2, axis=1)
    batch_result = 10*np.log10(up/down)
    print(batch_result)
    print('----------------------------------------------')
    return np.mean(batch_result)

def get_batch_MSE_OFDM(Hhat, H, Antenna):
    H = H.cpu().numpy()
    H_real = H[:,:int(Antenna)]
    H_imag = H[:,int(Antenna):]
    H = H_real + 1j*H_imag
    up = np.sum(abs(Hhat - H)**2, axis=1)
    down = np.sum(abs(H)**2, axis=1)
    batch_result = (up/down)
    print(batch_result)
    print('----------------------------------------------')
    return np.mean(batch_result)

def SE(H, H_hat, N):
    R_allcarrier_allsample=[]
    H = H.cpu().numpy()
    H_real = H[:,:32,:]
    H_imag = H[:,32:,:]
    H = H_real + 1j*H_imag
    for i in range(np.size(H,0)):
        for j in range(np.size(H,2)):
            H_hat_1 = H_hat[i,:,j]
            H_1 = H[i,:,j]
            H_hat_1 = np.expand_dims(H_hat_1, axis=1)
            H_1 = np.expand_dims(H_1, axis=1)
            H_hat_1 = H_hat_1 / np.linalg.norm(H_hat_1, 2)
            H_1 = H_1 / np.linalg.norm(H_1, 2)
            H_hat_1H = np.conj(np.transpose(H_hat_1))
            f = H_hat_1H
            f = f / np.linalg.norm(f, 2)
            SNR = np.linalg.norm(np.matmul(f, H_1), 2) ** 2 / N
            R = np.log2(1 + SNR)
            R_allcarrier_allsample.append(R)
    return np.mean(R_allcarrier_allsample)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NMSE_OFDM_ALL=[]
NMSE_SC_ALL=[]
NMSE_OFDM_ALL_v2=[]
NMSE_SC_ALL_v2=[]
SE_OFDM=[]
for SNR in range(0,21,3):
    Total_H_estimate=[]
    Total_H_label=[]
    Average_NMSE_scarrier=[]
    Average_NMSE_OFDM=[]
    Average_MSE_scarrier=[]
    Average_MSE_OFDM=[]
    Average_se_proposed=[]
    for simu in range(1):
        learning_rate = 1*1e-3  
        EPOCHS = 300  
        BATCH_SIZE = 32   
        Num_pilots = 16
        Num_Carriers = 64
        Num_user = 1
        Antenna = 32
        Angle_res = 64
        over = 2
        b_H_size = over * Antenna * 2
        H_size = Antenna * 2
        Length_LiDAR_feature = 128   # LiDAR feature
        Length_Signal_feature = 512  # Signal feature
        r=8 
        squzeenum = (Length_Signal_feature+Length_LiDAR_feature)/r
        Num_pilots_Fre = np.linspace(1,64,16,endpoint=False,dtype='int')


        net = model.LECEN_16p(Num_pilots,Length_Signal_feature,Length_LiDAR_feature,squzeenum,Num_user,b_H_size,H_size,over).to(device)
        # if pilot number is 32, then:
        # net = model.LECEN_32p(Num_pilots,Length_Signal_feature,Length_LiDAR_feature,squzeenum,Num_user,b_H_size,H_size,over).to(device)
        
        net = net.double()
        loss_func = nn.MSELoss()

        optimizer_m = torch.optim.Adam(net.parameters(), lr=learning_rate)  
        LR_sch = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, [80,120,150,180], gamma=0.3, last_epoch=-1)  


        train_input = torch.load('/data2/Haotiandata/Channel_Estimation/240412_Train_data_for_OFDM/T32/1RFchain/SNR'+str(SNR)+'/train_input_SNR'+str(SNR)+'Fpilot16T16.pth') 
        # if pilot number is 32, then:
        # train_input = torch.load('/data2/Haotiandata/Channel_Estimation/240412_Train_data_for_OFDM/T32/1RFchain/SNR'+str(SNR)+'/train_input_SNR'+str(SNR)+'Fpilot16T32.pth') 
        train_input_LiDAR = torch.load('/data2/Haotiandata/Channel_Estimation/240412_Train_data_for_OFDM/T32/train_input_lidar.pth.pth')  
        # label
        train_label_car5 = torch.load('/data2/Haotiandata/Channel_Estimation/240412_Train_data_for_OFDM/T32/1RFchain/SNR'+str(SNR)+'/train_label_SNR'+str(SNR)+'Fpilot16T16.pth')
        # if pilot number is 32, then:
        # train_label_car5 = torch.load('/data2/Haotiandata/Channel_Estimation/240412_Train_data_for_OFDM/T32/1RFchain/SNR'+str(SNR)+'/train_label_SNR'+str(SNR)+'Fpilot16T32.pth')
        train_sparse_label_car5 = torch.load('/data2/Haotiandata/Channel_Estimation/240412_Train_data_for_OFDM/T32/1RFchain/SNR'+str(SNR)+'/train_sparse_label_SNR'+str(SNR)+'Fpilot16T16.pth')
        # if pilot number is 32, then:
        # train_sparse_label_car5 = torch.load('/data2/Haotiandata/Channel_Estimation/240412_Train_data_for_OFDM/T32/1RFchain/SNR'+str(SNR)+'/train_sparse_label_SNR'+str(SNR)+'Fpilot16T32.pth')
        
        num_train = int(len(train_label_car5)*0.8)

        
        file_name1 = "Train_List.mat"
        file_name2 = "Test_List.mat"
        data1 = scipy.io.loadmat(file_name1)
        data2 = scipy.io.loadmat(file_name2)
        train_list = data1['T_L'][:]
        test_list = data2['V_L'][:]
        train_list = train_list.squeeze()
        test_list = test_list.squeeze()

        train_data = dataset_Multitemp.naiveDataset_nu([train_input[i] for i in train_list], [train_input_LiDAR[i] for i in train_list], [train_label_car5[i] for i in train_list], [train_sparse_label_car5[i] for i in train_list]    )
        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        val_data = dataset_Multitemp.naiveDataset_nu([train_input[i] for i in test_list], [train_input_LiDAR[i] for i in test_list], [train_label_car5[i] for i in test_list], [train_sparse_label_car5[i] for i in test_list])
        val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


        Loss=[]
        val_Loss=[]
        for time in range(1):
                min_val_loss = 100
                step_sum=[]
                loss_sum=[]
                count=0
                savetimes=0
                for i in range(EPOCHS):
                    net.train()
                    total_loss = 0
                    cnt = 0
                    for batch, (ipt, ipt_lidar, lb, sp_lb) in enumerate(train_dataloader):
                        count+=1
                        ipt = ipt.to(device)   
                        ipt_lidar = ipt_lidar.to(device)
                        ipt_lidar_n = scale_three_channels(ipt_lidar)
                        ipt_real = ipt.real
                        ipt_imag = ipt.imag
                        ipt_tmp = torch.cat((ipt_real, ipt_imag), dim=1).squeeze(2).to(device)
                        ipt_tmp1 = ipt_tmp[:,:,1,:]
                        lb = lb.to(device)
                        lb1 = lb[:,:,Num_pilots_Fre[1]]
                        output = net(ipt_tmp1,ipt_lidar_n)
                        loss = loss_func(output, lb1)
                        cnt += 1
                        total_loss += loss.item()
                        optimizer_m.zero_grad()
                        loss.backward()
                        optimizer_m.step()
                    loss_sum.append(total_loss/cnt)
                    step_sum.append(count)
                    LR_sch.step()
         

                    
                    net.eval()
                    total_loss_val = 0
                    cnt_val = 0
                    total_NMSE_val = 0
                    total_NMSE_val1 = 0
                    total_NMSE_val2 = 0
                    with torch.no_grad():
                        for batch, (ipt, ipt_lidar, lb, sp_lb) in enumerate(val_dataloader):
                            ipt = ipt.to(device)
                            ipt_real = ipt.real
                            ipt_imag = ipt.imag
                            ipt_tmp = torch.cat((ipt_real, ipt_imag), dim=1).squeeze(2).to(device)
                            ipt_tmp1 = ipt_tmp[:,:,1,:]
                            ipt_lidar = ipt_lidar.to(device)
                            ipt_lidar_n = scale_three_channels(ipt_lidar)
                            lb1 = lb[:,:,Num_pilots_Fre[1]]
                            lb1 = lb1.to(device)
                            
                            output = net(ipt_tmp1,ipt_lidar_n)
                            loss = loss_func(output, lb1)
                            cnt_val += 1
                            total_loss_val += loss.item()
                        val_Loss.append(total_loss_val/cnt_val)

                    print("EPOCH===========================%d-32pilot, %d"%(SNR,i))
                    print("simu:",simu)
                    print("Train Loss", total_loss / cnt)
                    print("Val Loss", total_loss_val / cnt_val)
                    print(total_loss_val / cnt_val < min_val_loss)
                    
                # evaluation
                net.eval()
                H_estimate=[]
                H_label=[]
                Val_Pilot_Channel=[]   
                Val_Channel_Label=[]
                with torch.no_grad():
                    total_loss_val = 0
                    cnt_val = 0
                    total_NMSE_val = 0
                    total_MSE_val = 0
                    total_NMSE_s = np.zeros(len(Num_pilots_Fre))
                    for batch, (ipt, ipt_lidar, lb, sp_lb) in enumerate(val_dataloader):
                        ipt = ipt.to(device)
                        ipt_lidar = ipt_lidar.to(device)
                        ipt_lidar_n = scale_three_channels(ipt_lidar)
                        ipt_real = ipt.real
                        ipt_imag = ipt.imag
                        ipt_tmp = torch.cat((ipt_real, ipt_imag), dim=1).squeeze(2).to(device)
                        ipt_tmp1 = ipt_tmp[:,:,1,:]
                        output1 = net(ipt_tmp1,ipt_lidar_n)
                        output1 = output1.cpu().numpy()
                        output1_real = output1[:,:int(Antenna*Num_user)]
                        output1_imag = output1[:,int(Antenna*Num_user):]
                        output1 = output1_real + 1j*output1_imag


                        cnt_val += 1
                        lb1 = lb[:,:,Num_pilots_Fre[1]]
                        total_NMSE_val += get_batch_NMSE2(output1, lb1, Antenna)
                        total_MSE_val += get_batch_MSE2(output1, lb1, Antenna)
                        Pilot_Channel = np.zeros([BATCH_SIZE,Antenna,Num_Carriers], dtype=complex)
                        for s in range(len(Num_pilots_Fre)):
                            ipt_s = ipt[:,:,s,:]
                            ipt_s = ipt_s.to(device)
                            ipt_s_real = ipt_s.real
                            ipt_s_imag = ipt_s.imag
                            ipt_s_tmp = torch.cat((ipt_s_real, ipt_s_imag), dim=1).squeeze(2).to(device)
                            output_s = net(ipt_s_tmp,ipt_lidar_n)
                            output_s = output_s.cpu().numpy()
                            output_s_real = output_s[:,:int(Antenna*Num_user)]
                            output_s_imag = output_s[:,int(Antenna*Num_user):]
                            output_s = output_s_real + 1j*output_s_imag
                            lb_s = lb[:,:,Num_pilots_Fre[s]]                       
                            total_NMSE_s[s] += get_batch_NMSE2(output_s, lb_s, Antenna)
                            Pilot_Channel[:,:,Num_pilots_Fre[s]] = output_s
                            print('%d-th subcarrier NMSE:%f' %(Num_pilots_Fre[s],total_NMSE_s[s]), 'dB')
                            print('----------------------------------------------')
                        Pilot_Channel = torch.from_numpy(Pilot_Channel)
                        Val_Pilot_Channel.append(Pilot_Channel)
                        Val_Channel_Label.append(lb)
                        print('************************************************')
                        print(total_NMSE_s/cnt_val) 
                        
                        H_estimate.append(output1)
                        lb = lb.cpu().numpy()
                        lb1 = lb1.cpu().numpy()
                        lb1_real = lb1[:, :int(Antenna)]
                        lb1_imag = lb1[:, int(Antenna):]
                        h_lb1 = lb1_real + 1j * lb1_imag
                        h_lb1 = h_lb1.transpose(1,0)
                        H_label.append(h_lb1)
                    
                    Val_Pilot_Channel = torch.cat(Val_Pilot_Channel, dim=0)  
                    Val_Channel_Label = torch.cat(Val_Channel_Label, dim=0)   
                    print("NMSE: ", total_NMSE_val/ cnt_val, 'dB')
                    Average_NMSE_scarrier.append(total_NMSE_val/ cnt_val)
                    Average_MSE_scarrier.append(total_MSE_val/ cnt_val)
                    print(Average_NMSE_scarrier)

        Total_H_estimate.append(H_estimate)      
        Total_H_label.append(H_label) 
  
        #### CSI reconsturction at all subcarriers #####
        # load CI-CNN
        Inter_model = torch.load("/RecoverNet_Pilot8_T32_5_to_20.pth", map_location = device)
        OFDM_NMSE_val = 0
        Inter_model.eval()
        Num_pilots_Fre_using = np.linspace(1,Num_Carriers,8,endpoint=False,dtype='int')

        mask = torch.zeros_like(Val_Pilot_Channel) 
        mask[:, :,  Num_pilots_Fre_using] = 1        
        Val_Pilot_Channel = mask*Val_Pilot_Channel
        

        #####data processing#####
        Val_Pilot_Channel = Val_Pilot_Channel.to(device)         
        Val_Pilot_Channel_real = Val_Pilot_Channel.real
        Val_Pilot_Channel_imag = Val_Pilot_Channel.imag
        Val_Pilot_Channel_angle = torch.angle(Val_Pilot_Channel)
        Val_Pilot_Channel_tmp = torch.stack((Val_Pilot_Channel_real, Val_Pilot_Channel_imag, Val_Pilot_Channel_angle), dim=1).to(device)
        OFDM_Channel = Inter_model(Val_Pilot_Channel_tmp)
        
        OFDM_Channel = OFDM_Channel.cpu().detach().numpy()
        OFDM_Channel_real = OFDM_Channel[:,:int(Antenna),:]
        OFDM_Channel_imag = OFDM_Channel[:,int(Antenna):,:]
        OFDM_Channel = OFDM_Channel_real + 1j*OFDM_Channel_imag
        OFDM_NMSE_val =  get_batch_NMSE_OFDM(OFDM_Channel, Val_Channel_Label, Antenna)
        OFDM_NMSE_val_mse =  get_batch_MSE_OFDM(OFDM_Channel, Val_Channel_Label, Antenna)
        
        opt = OFDM_Channel
        label = Val_Channel_Label.numpy()
        
        print("OFDM recover accuracy: %fdB"%(OFDM_NMSE_val))
        print('------------------------------------------')
        print("OFDM recover accuracy: %fdB"%(10*np.log10(OFDM_NMSE_val_mse)))
        Average_NMSE_OFDM.append(OFDM_NMSE_val)
        Average_MSE_OFDM.append(OFDM_NMSE_val_mse)
        #SE
        sigma = 1 / (10 ** (SNR / 10))  
        se_proposed = SE(Val_Channel_Label,OFDM_Channel,sigma)   
        Average_se_proposed.append(se_proposed)
        print("Average SE: %f bps/Hz"%(se_proposed))
        print('---------------------------------------------------')
        print("SNR%d--Time%d's NMSE single carrier %f"%(SNR,simu,  10*np.log10(np.mean(Average_MSE_scarrier))))
        print("SNR%d--Time%d's NMSE OFDM %f"%(SNR,simu, 10*np.log10(np.mean(Average_MSE_OFDM))))
        print("SNR%d--Time%d's SE %f"%(SNR,simu,np.mean(Average_se_proposed)))


    print("SNR%d--single carrier NMSE(100 times):%f "%(SNR, 10*np.log10(np.mean(Average_MSE_scarrier))), 'dB')
    print("SNR%d--OFDM NMSE(100 times):%f "%(SNR, 10*np.log10(np.mean(Average_MSE_OFDM))), 'dB')
    NMSE_OFDM_ALL.append(np.mean(Average_NMSE_OFDM))
    NMSE_SC_ALL.append(np.mean(Average_NMSE_scarrier))
    #mse
    NMSE_OFDM_ALL_v2.append(10*np.log10(np.mean(Average_MSE_OFDM)))
    NMSE_SC_ALL_v2.append(10*np.log10(np.mean(Average_MSE_scarrier)))
    #se
    SE_OFDM.append(np.mean(Average_se_proposed))

    print('ALL carrier:')
    print(NMSE_OFDM_ALL_v2)
    print(SE_OFDM)
    print('Single carrier:')
    print(NMSE_SC_ALL_v2)
