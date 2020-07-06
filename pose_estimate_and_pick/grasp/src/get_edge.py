import numpy as np
from scipy import signal 


def get_range(im_test): #藉由mask反推所有物體邊界 ->拿來切跟壓縮用
    #for 測edge ，換兩方向壓成一維
    mode = {1:"merge",2:"merge",3:"merge",4:"divide",5:"divide"}
    
    #mode = "merge"
    
    class_m = np.unique(im_test) #確定類數
    print(class_m[1:])
    classwise_m = np.zeros((im_test.shape[0],im_test.shape[1],class_m.shape[0]))
    
    edge = np.zeros((4,2,3)) #左右上下
    sub_edge = [] #sub_edge.append([1,2])即可
    
    label = [1,2,3] #之後要append 4,5在後
    
    in_cla = 1 
    while in_cla < class_m.shape[0]:
        print(in_cla)
        classwise_m[:,:,in_cla] = np.where(im_test==class_m[in_cla],1,0)#切割各class
    
    #area = np.sum(test_h)
        if mode[in_cla] == "divide":
            im_c=signal.convolve2d(classwise_m[:,:,in_cla], np.ones((40,40)), boundary='symm', mode='same')
            im_e2=np.where(im_c>np.percentile(im_c, 70),1,0)#跟侵蝕不同，是根據比例侵蝕
            
            test_h2 = np.where(np.sum(im_e2,axis=0)>0,1,0) #垂直方向往下加 , (test.shape[1])
            test_v2 = np.where(np.sum(im_e2,axis=1)>0,1,0) #水平方向往旁加 , (test.shape[0])
        
            test_hb2 = test_h[1:]-test_h[0:(im_e2.shape[1]-1)]
            test_vb2 = test_v[1:]-test_v[0:(im_e2.shape[0]-1)]
            
        test_h = np.where(np.sum(classwise_m[:,:,in_cla],axis=0)>0,1,0) #垂直方向往下加 , (test.shape[1])
        test_v = np.where(np.sum(classwise_m[:,:,in_cla],axis=1)>0,1,0) #水平方向往旁加 , (test.shape[0])
        
        test_hb = test_h[1:]-test_h[0:(classwise_m[:,:,in_cla].shape[1]-1)]
        test_vb = test_v[1:]-test_v[0:(classwise_m[:,:,in_cla].shape[0]-1)]
                
        if mode[in_cla] == "merge":
            if np.sum(np.abs(test_hb))>2:
                edge[0,1,in_cla-1] = np.argmax(test_hb*np.arange(test_hb.shape[0],0,-1))+1 
                edge[0,0,in_cla-1] = np.argmax(classwise_m[:,int(edge[0,1,in_cla-1]),in_cla])
                
                edge[1,1,in_cla-1] = np.argmin(test_hb*np.arange(0,test_hb.shape[0],1))       
                edge[1,0,in_cla-1] = np.argmax(classwise_m[:,int(edge[1,1,in_cla-1]),in_cla])
            
            else:
                edge[0,1,in_cla-1] = np.argmax(test_hb)+1 #水平方向 的左bound
                edge[0,0,in_cla-1] = np.argmax(classwise_m[:,int(edge[0,1,in_cla-1]),in_cla]) #找到有值點->左bound 高
                #左bound 的座標
                                                         
                edge[1,1,in_cla-1] = np.argmin(test_hb) 
                edge[1,0,in_cla-1] = np.argmax(classwise_m[:,int(edge[1,1,in_cla-1]),in_cla])
                #右bound 的座標
                                                           
            if np.sum(np.abs(test_vb))>2:
                edge[2,0,in_cla-1] = np.argmax(test_vb*np.arange(test_vb.shape[0],0,-1))+1
                edge[2,1,in_cla-1] = np.argmax(classwise_m[int(edge[2,0,in_cla-1]),:,in_cla])
                
                edge[3,0,in_cla-1] = np.argmin(test_vb*np.arange(0,test_vb.shape[0],1)) 
                edge[3,1,in_cla-1] = np.argmax(classwise_m[int(edge[3,0,in_cla-1]),:,in_cla])
                
            else:
                edge[2,0,in_cla-1] = np.argmax(test_vb)+1#上bound
                edge[2,1,in_cla-1] = np.argmax(classwise_m[int(edge[2,0,in_cla-1]),:,in_cla]) #找到有值點->左bound 高
                #左bound 的座標
                                                         
                edge[3,0,in_cla-1] = np.argmin(test_vb)#下bound
                edge[3,1,in_cla-1] = np.argmax(classwise_m[int(edge[3,0,in_cla-1]),:,in_cla])
                
        else:#mode[in_cla] ==divde
            edge_cla = np.zeros((4,2))
            h_index = np.argsort(test_hb2*np.arange(test_hb2.shape[0],0,-1))       
            #左bound記得加1 +1
            #h_index[i]+1 ->左bound座標 h_index[-(i+1)]->右bound座標
            #i.e h_index+1,h_index[::-1]
            
            v_index = np.argsort(test_vb2*np.arange(test_vb2.shape[0],0,-1))
            #上bound記得加1 +1       
            max_pro_num = int(np.sum(np.abs(test_vb2))//2)*(np.sum(np.abs(test_hb2))//2) #最大候選人數
            
            for ind in range(max_pro_num):
                if (im_e2[(h_index[ind]+1+h_index[::-1][ind])//2,(v_index[ind]+1+v_index[::-1][ind])//2]):
                    edge_cla[0,1]=h_index[ind]+1 #sub_edge.append([1,2])即可
                    edge_cla[0,0]=np.argmax(im_e2[:,h_index[ind]+1])
                    
                    edge_cla[1,1]=h_index[::-1][ind] #sub_edge.append([1,2])即可
                    edge_cla[1,0]=np.argmax(im_e2[:,h_index[::-1][ind]])
                    
                    edge_cla[2,0]=v_index[ind]+1 #sub_edge.append([1,2])即可
                    edge_cla[2,1]=np.argmax(im_e2[v_index[ind]+1,:])
                    
                    edge_cla[3,0]=v_index[::-1][ind] #sub_edge.append([1,2])即可
                    edge_cla[3,1]=np.argmax(im_e2[v_index[::-1][ind],:])                                               
                    
                    sub_edge.append(edge_cla)
                    sub_edge = np.transpose(np.array(sub_edge),[1,2,0])
                    label.append(in_cla) #之後要append 4,5在後 
                    
            #im_e2[]==in_cla ->輸出true and false
            #上bound記得加1 +1
            
            #同類要切割成不同物體
        
        in_cla += 1   
    #edge = np.append(edge,sub_edge,axis=2)      
    #edge[4] = area
    
    return edge.astype(int),label#產生四個頂點 - >用遠近中等的兩點求斜率，(最遠是對角)
                                     #樂高 -> 距離長兩點求斜率
                           #四個點的平均位置就是pose之location
def pixel_tr(pixel,deck_pose,edge_deck):
    real_pose = np.zeros(3)#x,y,z 三軸 
    real_pose[2] = np.mean(deck_pose[:,2])
    real_pose[0] = deck_pose[0,0]+(deck_pose[1,0]-deck_pose[0,0])*(pixel[0]-edge_deck[0,0]/(edge_deck[1,0]-edge_deck[0,0])) #左右 插值
    
    real_pose[1] = deck_pose[2,1]+(deck_pose[3,1]-deck_pose[2,1])*(pixel[1]-edge_deck[2,1]/(edge_deck[3,1]-edge_deck[2,1])) #上下 插值
    
    return real_pose
    
def pose_em(edge_ob,deck_pose,edge_deck,label):#deck_pose
    #deck_pose  #(x,y,z)高度確定(必同)，可以用平均
    loca = pixel_tr(np.mean(edge_ob,axis=0),deck_pose,edge_deck)
    #if label<=3:    
     #   ori =  #x,y,z,w
        
    #else:# 為4,5類

def l_get(pixel):
    position = np.zeros(2)
    position[0] = (pixel[0]-555)/-839 
    position[1] = (pixel[1]-354)/-1071
    
    return position #x,y
    